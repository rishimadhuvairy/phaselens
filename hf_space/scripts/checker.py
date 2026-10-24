# scripts/spectrum_generator and scripts/peak_finder are extensionless.
# This file uses a .py suffix so it can be run as a module (python -m scripts.checker).
"""Check a peak assignment against the mode card and the spectrum context.

The checker reads the mode card named by ``context.mode_card`` and the
context metadata (substrate and known artifact lines). It does not read
``answers`` or any other gold labels.

Rules, applied in this order for each input peak:

- The parsed list must pair every input peak, keep ``position_cm-1``, and
  carry exactly one label. Pairing allows 0.05 cm^-1 so a trivial echo of
  the printed position still joins. The mode window below is a separate 4.0.
- The label is ``artifact`` or one symmetry name on that card.
  ``canonical_label`` makes ``A1'``, primes, and ``A1g + B1g`` spacing match.
- A silent mode listed on the card cannot be used.
- A peak farther than 4.0 cm^-1 from a mode's card frequency cannot take
  that mode's name. 4.0 cm^-1 is the generator's match tolerance. If several
  card modes sit inside the window, the label is accepted when it is one of
  those modes.
- A peak within 4.0 cm^-1 of a plasma line in
  ``context.known_artifact_lines`` (or ``known_artifact_lines_cm-1``) must
  be ``artifact``.
- A peak within 4.0 cm^-1 of the context silicon line must be ``artifact``
  only when that line is present and the sample is not silicon. The silicon
  card's own F2g line stays a real mode. There is no global "near 520"
  rule: anatase's combination band sits near 517.

Run from the repo root::

    py -3.12 -m scripts.checker --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_str = str(_REPO_ROOT)
sys.path[:] = [_repo_str, *[p for p in sys.path if p != _repo_str]]

from scripts.label_norm import canonical_label  # noqa: E402
from scripts.qwen_config import ARTIFACT_LABEL, IGNORED_LABEL, MODE_CARDS_DIR  # noqa: E402

# Same numeric window the spectrum generator uses when it joins a found peak
# to the peak it planted (scripts/spectrum_generator, ``best_d <= 4.0``).
FREQUENCY_TOLERANCE_CM = 4.0
# Echo tolerance for "preserve position_cm-1". This is not the mode window.
POSITION_PAIR_TOLERANCE_CM = 0.05

_SKIP_KEY_WORDS = ("silent", "artifact", "instrument", "unobserved", "substrate", "plasma")
_POSITION_KEYS = ("position_cm-1", "position_cm1", "position")
_INACTIVE_ACTIVITY = {"second-order", "silent", "inactive"}


@dataclass(frozen=True)
class ModeFreq:
    name: str
    position: float


@dataclass(frozen=True)
class CardRules:
    modes: tuple[ModeFreq, ...]
    allowed: frozenset[str]
    silent: frozenset[str]


@dataclass
class CheckOutcome:
    passed: bool
    reasons: list[str]
    adjusted_labels: list[dict]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _key_skipped(key: str) -> bool:
    low = key.casefold()
    return any(word in low for word in _SKIP_KEY_WORDS)


def _is_silent_key(key: str) -> bool:
    return "silent" in key.casefold()


def symmetry_tokens(text: object) -> list[str]:
    """Split a card symmetry field into canonical names.

    ``E' (monolayer); E1_2g (bulk notation)`` yields both names.
    ``A1g + B1g`` stays one name. A trailing word such as ``shear`` is dropped.
    """
    if not isinstance(text, str):
        return []
    tokens: list[str] = []
    for piece in text.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        piece = re.split(r"\s+\(", piece, maxsplit=1)[0].strip()
        parts = piece.split()
        if len(parts) >= 3 and parts[1] == "+":
            token = " ".join(parts[:3])
        elif parts:
            token = parts[0]
        else:
            continue
        token = canonical_label(token)
        if token and token.casefold() != "null":
            tokens.append(token)
    return tokens


def _walk_active(obj: object):
    if isinstance(obj, dict):
        yield obj
        for key, value in obj.items():
            if isinstance(key, str) and _key_skipped(key):
                continue
            yield from _walk_active(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_active(value)


def _collect_silent(obj: object, under_silent: bool = False, found: list[str] | None = None) -> list[str]:
    if found is None:
        found = []
    if isinstance(obj, dict):
        if under_silent:
            found.extend(symmetry_tokens(obj.get("symmetry")))
        for key, value in obj.items():
            if not isinstance(key, str):
                _collect_silent(value, under_silent, found)
                continue
            child_silent = under_silent or _is_silent_key(key)
            if child_silent and isinstance(value, str):
                found.extend(symmetry_tokens(value))
            elif child_silent and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        found.extend(symmetry_tokens(item))
            _collect_silent(value, child_silent, found)
    elif isinstance(obj, list):
        for value in obj:
            _collect_silent(value, under_silent, found)
    return found


def _collect_raman_active_names(obj: object, found: list[str] | None = None) -> list[str]:
    """Names listed as Raman-active, even when no frequency is attached."""
    if found is None:
        found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            if _is_silent_key(key) or _key_skipped(key):
                continue
            if "raman_active" in key.casefold() and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        found.extend(symmetry_tokens(item))
            else:
                _collect_raman_active_names(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _collect_raman_active_names(value, found)
    return found


def _position_of_dict(item: dict) -> float | None:
    for key in _POSITION_KEYS:
        if key not in item:
            continue
        number = _number(item.get(key))
        if number is not None:
            return number
    return None


def card_rules(card: dict) -> CardRules:
    """Active (name, frequency) pairs, allowed names, and silent names.

    Silent, artifact, instrument, substrate, and plasma sections are not
    active modes. ``training_eligible: false``, inactive Raman activity, and
    ``raman_active: false`` are skipped, matching the spectrum generator's
    usable-mode filter. Positions are not collapsed, so two symmetries a few
    cm^-1 apart both stay available inside the 4 cm^-1 window.
    """
    if not isinstance(card, dict):
        raise TypeError("mode card must be a JSON object")
    modes: list[ModeFreq] = []
    allowed: set[str] = set()
    for item in _walk_active(card):
        if item.get("training_eligible") is False:
            continue
        if item.get("raman_active") is False:
            continue
        activity = item.get("raman_activity")
        if isinstance(activity, str) and activity.casefold() in _INACTIVE_ACTIVITY:
            continue
        names = symmetry_tokens(item.get("symmetry"))
        if not names:
            continue
        allowed.update(names)
        position = _position_of_dict(item)
        if position is None:
            continue
        for name in names:
            modes.append(ModeFreq(name, position))
    allowed.update(_collect_raman_active_names(card))
    silent = set(_collect_silent(card))
    allowed -= silent
    # Stable unique modes.
    unique: list[ModeFreq] = []
    seen: set[tuple[str, float]] = set()
    for mode in modes:
        if mode.name in silent:
            continue
        key = (mode.name, round(mode.position, 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append(mode)
    return CardRules(tuple(unique), frozenset(allowed), frozenset(silent))


def load_card(mode_card: object, cards_dir: Path = MODE_CARDS_DIR) -> dict:
    if not isinstance(mode_card, str) or not mode_card.strip():
        raise SystemExit("spectrum context.mode_card is missing")
    name = Path(mode_card).name
    if name != mode_card or ".." in name:
        raise SystemExit(f"refusing mode card path {mode_card!r}")
    path = cards_dir / name
    if not path.is_file():
        raise SystemExit(f"mode card not found: {path}")
    card = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise SystemExit(f"mode card is not a JSON object: {path}")
    return card


_CARD_CACHE: dict[str, CardRules] = {}


def rules_for(mode_card: object, cards_dir: Path = MODE_CARDS_DIR) -> CardRules:
    if not isinstance(mode_card, str) or not mode_card.strip():
        raise SystemExit("spectrum context.mode_card is missing")
    name = Path(mode_card).name
    cached = _CARD_CACHE.get(name)
    if cached is not None:
        return cached
    rules = card_rules(load_card(name, cards_dir))
    _CARD_CACHE[name] = rules
    return rules


def artifact_block(context: dict) -> dict:
    if not isinstance(context, dict):
        return {}
    for key in ("known_artifact_lines", "known_artifact_lines_cm-1"):
        block = context.get(key)
        if isinstance(block, dict):
            return block
    return {}


def plasma_lines(context: dict) -> list[float]:
    raw = artifact_block(context).get("plasma")
    if isinstance(raw, list):
        return [number for item in raw if (number := _number(item)) is not None]
    number = _number(raw)
    if number is None:
        return []
    return [number]


def silicon_line(context: dict) -> float | None:
    return _number(artifact_block(context).get("silicon"))


def card_is_silicon(mode_card: object) -> bool:
    if not isinstance(mode_card, str):
        return False
    stem = Path(mode_card).name.casefold().removesuffix(".json")
    return stem == "si" or stem == "silicon" or stem.startswith("silicon_")


def material_is_silicon(material: object) -> bool:
    if not isinstance(material, str):
        return False
    text = material.casefold()
    if "disulfide" in text or "mos2" in text or "mos₂" in text:
        return False
    return "silicon" in text or "si-i" in text or "diamond-cubic" in text


def _is_artifact_label(label: str) -> bool:
    return canonical_label(label).casefold() == ARTIFACT_LABEL.casefold()


def _fmt_cm(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _nearby(rules: CardRules, peak: float) -> list[ModeFreq]:
    return [mode for mode in rules.modes if abs(mode.position - peak) <= FREQUENCY_TOLERANCE_CM]


def _label_value(item: dict) -> object:
    for key in ("label", "symmetry", "assignment"):
        if key in item and item[key] is not None:
            return item[key]
    return None


def check_assignment(
    peaks: list,
    parsed_labels: list | None,
    card: dict | CardRules,
    context: dict,
) -> CheckOutcome:
    """Return reasons and one adjusted label per input peak.

    A peak that fails is adjusted to ``unassigned``. A peak that passes keeps
    the model's label string. This function never reads gold answers.
    """
    rules = card if isinstance(card, CardRules) else card_rules(card)
    if not isinstance(context, dict):
        context = {}
    reasons: list[str] = []
    input_peaks: list[tuple[float, dict]] = []
    if not isinstance(peaks, list):
        peaks = []
    for index, peak in enumerate(peaks):
        if not isinstance(peak, dict):
            reasons.append(f"input peak {index} is not an object")
            continue
        position = _position_of_dict(peak)
        if position is None:
            reasons.append(f"input peak {index} has no position_cm-1")
            continue
        input_peaks.append((position, peak))

    parsed: list | None
    if parsed_labels is None:
        parsed = None
        reasons.append("model output is not a JSON list pairing each peak with one label")
    elif not isinstance(parsed_labels, list):
        parsed = None
        reasons.append("model output is not a JSON list pairing each peak with one label")
    else:
        parsed = parsed_labels

    # peak index -> list of parsed indexes within the echo tolerance
    owners: list[list[int]] = [[] for _ in input_peaks]
    extras: list[int] = []
    if parsed is not None:
        for parsed_index, item in enumerate(parsed):
            if not isinstance(item, dict):
                reasons.append(f"parsed item {parsed_index} is not an object")
                extras.append(parsed_index)
                continue
            position = _position_of_dict(item)
            if position is None:
                reasons.append(f"parsed item {parsed_index} has no position_cm-1")
                extras.append(parsed_index)
                continue
            best: int | None = None
            best_distance: float | None = None
            for peak_index, (peak_position, _peak) in enumerate(input_peaks):
                distance = abs(position - peak_position)
                if distance <= POSITION_PAIR_TOLERANCE_CM and (
                    best_distance is None or distance < best_distance
                ):
                    best = peak_index
                    best_distance = distance
            if best is None:
                extras.append(parsed_index)
                reasons.append(
                    f"parsed position {_fmt_cm(position)} cm^-1 does not match an input peak"
                )
            else:
                owners[best].append(parsed_index)

    mode_card = context.get("mode_card")
    silicon = silicon_line(context)
    force_silicon = (
        silicon is not None
        and not material_is_silicon(context.get("material"))
        and not card_is_silicon(mode_card)
    )
    plasma = plasma_lines(context)

    adjusted: list[dict] = []
    for peak_index, (peak_position, _peak) in enumerate(input_peaks):
        peak_reasons: list[str] = []
        where = f"peak at {_fmt_cm(peak_position)} cm^-1"
        chosen: dict | None = None
        if parsed is None:
            peak_reasons.append(f"{where} has no label")
        else:
            owned = owners[peak_index]
            if not owned:
                peak_reasons.append(f"{where} has no label")
            elif len(owned) > 1:
                peak_reasons.append(f"{where} has {len(owned)} labels; one peak must have one label")
            else:
                chosen = parsed[owned[0]] if isinstance(parsed[owned[0]], dict) else None
        label_text: str | None = None
        if chosen is not None and not peak_reasons:
            raw_label = _label_value(chosen)
            if isinstance(raw_label, list):
                peak_reasons.append(f"{where} has {len(raw_label)} labels; one peak must have one label")
            elif not isinstance(raw_label, str) or not raw_label.strip():
                peak_reasons.append(f"{where} has no label")
            else:
                label_text = raw_label.strip()
                canon = canonical_label(label_text)
                if canon in rules.silent:
                    peak_reasons.append(
                        f"{where} uses silent mode {canon}, which the card lists as silent"
                    )
                elif not _is_artifact_label(label_text) and canon not in rules.allowed:
                    peak_reasons.append(
                        f"{where} label {label_text} is not artifact and is not a symmetry on this mode card"
                    )
                elif not _is_artifact_label(label_text):
                    distances = [
                        abs(mode.position - peak_position)
                        for mode in rules.modes
                        if mode.name == canon
                    ]
                    if not distances or min(distances) > FREQUENCY_TOLERANCE_CM:
                        nearby = _nearby(rules, peak_position)
                        if nearby:
                            listed = ", ".join(
                                f"{mode.name} at {_fmt_cm(mode.position)}" for mode in nearby[:8]
                            )
                            window = f"Modes within {FREQUENCY_TOLERANCE_CM:.1f} cm^-1: {listed}."
                        else:
                            window = (
                                f"No card mode is within {FREQUENCY_TOLERANCE_CM:.1f} cm^-1 of this peak."
                            )
                        if not distances:
                            why = f"the card has no frequency for {canon}"
                        else:
                            why = (
                                f"the nearest card {canon} is {min(distances):.2f} cm^-1 away"
                            )
                        peak_reasons.append(
                            f"{where} cannot take {canon}: {why} (limit {FREQUENCY_TOLERANCE_CM:.1f}). {window}"
                        )
                if label_text is not None and not _is_artifact_label(label_text):
                    for line in plasma:
                        if abs(peak_position - line) <= FREQUENCY_TOLERANCE_CM:
                            peak_reasons.append(
                                f"{where} is within {FREQUENCY_TOLERANCE_CM:.1f} cm^-1 of plasma line "
                                f"{_fmt_cm(line)} and must be artifact, not {canonical_label(label_text)}"
                            )
                            break
                    if (
                        force_silicon
                        and silicon is not None
                        and abs(peak_position - silicon) <= FREQUENCY_TOLERANCE_CM
                    ):
                        peak_reasons.append(
                            f"{where} is within {FREQUENCY_TOLERANCE_CM:.1f} cm^-1 of the silicon "
                            f"substrate line at {_fmt_cm(silicon)} and must be artifact, not "
                            f"{canonical_label(label_text)}"
                        )
                elif label_text is not None and _is_artifact_label(label_text):
                    # Artifact is allowed off a plasma or silicon line. The
                    # scorer, not the checker, decides whether that was right.
                    pass
        reasons.extend(peak_reasons)
        if peak_reasons or label_text is None:
            adjusted.append({"position_cm-1": peak_position, "label": IGNORED_LABEL})
        else:
            adjusted.append({"position_cm-1": peak_position, "label": label_text})

    # De-duplicate reasons while keeping order (a structural note can repeat).
    unique_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for reason in reasons:
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        unique_reasons.append(reason)
    return CheckOutcome(passed=not unique_reasons, reasons=unique_reasons, adjusted_labels=adjusted)


def retry_prompt(original_prompt: str, reasons: list[str]) -> str:
    """Original prompt plus the checker reasons. No gold labels are added."""
    lines = "\n".join(f"- {reason}" for reason in reasons)
    return (
        original_prompt.rstrip()
        + "\n\nThe previous answer failed these checks:\n"
        + lines
        + "\n\nFix the JSON list. Pair every input peak, preserve position_cm-1, "
        + 'and give each peak one label: a symmetry name from the card, or "artifact". '
        + "Return only that JSON list.\n"
    )


def _expect(failures: list[str], outcome: CheckOutcome, passed: bool, name: str) -> None:
    if outcome.passed is not passed:
        failures.append(f"{name}: passed={outcome.passed} reasons={outcome.reasons}")


def self_test() -> int:
    failures: list[str] = []
    anatase = {
        "curated_expected_modes": [
            {"symmetry": "Eg", "position_cm-1": 145, "training_eligible": True},
            {"symmetry": "A1g + B1g", "position_cm-1": 517, "training_eligible": True},
            {"symmetry": "A1g", "position_cm-1": 512, "training_eligible": True},
            {"symmetry": "B1g", "position_cm-1": 518, "training_eligible": True},
            {"symmetry": "B1g", "position_cm-1": 396, "training_eligible": True},
        ],
        "symmetry_validation": {
            "literature_symmetry_table": {
                "raman_active": ["A1g", "B1g", "Eg"],
                "silent_modes": ["B2u"],
            }
        },
        "artifact_lines": {
            "substrate_raman": [{"symmetry": "F2g", "position_cm-1": 520.2, "training_eligible": True}]
        },
    }
    anatase_context = {
        "material": "anatase (TiO2)",
        "mode_card": "anatase_simplified.json",
        "substrate": None,
        "known_artifact_lines_cm-1": {"silicon": None, "plasma": [183.0]},
    }
    rules = card_rules(anatase)
    if "F2g" in rules.allowed:
        failures.append("substrate F2g was treated as an anatase mode")
    if "B2u" not in rules.silent or "B2u" in rules.allowed:
        failures.append(f"B2u silent set failed: {rules.silent}")
    if "A1g + B1g" not in rules.allowed:
        failures.append("combination band missing from allowed names")

    peaks = [{"position_cm-1": 514.25}, {"position_cm-1": 183.1}, {"position_cm-1": 145.4}]
    good = [
        {"position_cm-1": 514.25, "label": "A1g+B1g"},
        {"position_cm-1": 183.1, "label": "artifact"},
        {"position_cm-1": 145.4, "label": "Eg"},
    ]
    outcome = check_assignment(peaks, good, anatase, anatase_context)
    _expect(failures, outcome, True, "anatase combination and plasma")
    if outcome.adjusted_labels[0]["label"] != "A1g+B1g":
        failures.append("checker rewrote a passing label")

    near_517 = check_assignment(
        [{"position_cm-1": 516.8}],
        [{"position_cm-1": 516.8, "label": "A1g + B1g"}],
        anatase,
        anatase_context,
    )
    _expect(failures, near_517, True, "anatase near 517 is not forced to artifact")

    several = check_assignment(
        [{"position_cm-1": 515.5}],
        [{"position_cm-1": 515.5, "label": "B1g"}],
        anatase,
        anatase_context,
    )
    _expect(failures, several, True, "several modes in the window")

    far = check_assignment(
        [{"position_cm-1": 514.25}],
        [{"position_cm-1": 514.25, "label": "Eg"}],
        anatase,
        anatase_context,
    )
    _expect(failures, far, False, "Eg far from 145")
    if far.adjusted_labels[0]["label"] != IGNORED_LABEL:
        failures.append("failed peak was not left unassigned")

    silent = check_assignment(
        [{"position_cm-1": 145.0}],
        [{"position_cm-1": 145.0, "label": "B2u"}],
        anatase,
        anatase_context,
    )
    _expect(failures, silent, False, "silent B2u")

    not_on_card = check_assignment(
        [{"position_cm-1": 516.0}],
        [{"position_cm-1": 516.0, "label": "F2g"}],
        anatase,
        anatase_context,
    )
    _expect(failures, not_on_card, False, "F2g is not an anatase mode")

    plasma_miss = check_assignment(
        [{"position_cm-1": 183.1}],
        [{"position_cm-1": 183.1, "label": "Eg"}],
        anatase,
        anatase_context,
    )
    _expect(failures, plasma_miss, False, "plasma must be artifact")

    mos2 = {
        "curated_expected_modes": [
            {"symmetry": "E'", "position_cm-1": 384, "training_eligible": True},
            {"symmetry": "A1'", "position_cm-1": None, "training_eligible": False},
        ],
        "symmetry_validation": {
            "literature_symmetry_table": {
                "bulk_2H_silent": ["B1u"],
                "monolayer_raman_active": ["A1'", "E'", "E''"],
            }
        },
    }
    mos2_context = {
        "material": "molybdenum disulfide",
        "mode_card": "mos2.json",
        "substrate": "Si/SiO2",
        "known_artifact_lines_cm-1": {"silicon": 520.2, "plasma": []},
    }
    substrate_ok = check_assignment(
        [{"position_cm-1": 520.5}, {"position_cm-1": 384.2}],
        [{"position_cm-1": 520.5, "label": "artifact"}, {"position_cm-1": 384.2, "label": "E'"}],
        mos2,
        mos2_context,
    )
    _expect(failures, substrate_ok, True, "mos2 silicon substrate artifact")
    substrate_bad = check_assignment(
        [{"position_cm-1": 520.5}],
        [{"position_cm-1": 520.5, "label": "E'"}],
        mos2,
        mos2_context,
    )
    _expect(failures, substrate_bad, False, "mos2 silicon line labeled E'")
    prime = check_assignment(
        [{"position_cm-1": 384.0}],
        [{"position_cm-1": 384.0, "label": "E\u2032"}],
        mos2,
        mos2_context,
    )
    _expect(failures, prime, True, "unicode prime matches E'")

    silicon_card = {
        "curated_expected_modes": [
            {"symmetry": "F2g", "position_cm-1": 520.2, "training_eligible": True}
        ]
    }
    silicon_context = {
        "material": "pristine diamond-cubic Si-I",
        "mode_card": "silicon.json",
        "known_artifact_lines_cm-1": {"silicon": 520.2, "plasma": [495.2]},
    }
    silicon_ok = check_assignment(
        [{"position_cm-1": 519.72}, {"position_cm-1": 494.95}],
        [{"position_cm-1": 519.72, "label": "F2g"}, {"position_cm-1": 494.95, "label": "artifact"}],
        silicon_card,
        silicon_context,
    )
    _expect(failures, silicon_ok, True, "silicon card F2g is a real mode")

    missing = check_assignment(
        [{"position_cm-1": 10.0}, {"position_cm-1": 20.0}],
        [{"position_cm-1": 10.0, "label": "Eg"}],
        anatase,
        anatase_context,
    )
    _expect(failures, missing, False, "missing peak")
    if missing.adjusted_labels[1]["label"] != IGNORED_LABEL:
        failures.append("missing peak was not unassigned")

    duplicate = check_assignment(
        [{"position_cm-1": 145.0}],
        [
            {"position_cm-1": 145.0, "label": "Eg"},
            {"position_cm-1": 145.01, "label": "B1g"},
        ],
        anatase,
        anatase_context,
    )
    _expect(failures, duplicate, False, "two labels for one peak")

    unparsed = check_assignment([{"position_cm-1": 145.0}], None, anatase, anatase_context)
    _expect(failures, unparsed, False, "unparsed")
    if unparsed.adjusted_labels[0]["label"] != IGNORED_LABEL:
        failures.append("unparsed peak was not unassigned")

    # Real cards, when this repo layout is present.
    anatase_path = MODE_CARDS_DIR / "anatase_simplified.json"
    if anatase_path.is_file():
        real = card_rules(load_card("anatase_simplified.json"))
        if "A1g + B1g" not in real.allowed:
            failures.append("real anatase card missing A1g + B1g")
        if "F2g" in real.allowed:
            failures.append("real anatase card includes substrate F2g")
        if "B2u" not in real.silent:
            failures.append("real anatase card missing silent B2u")
        real_peak = check_assignment(
            [{"position_cm-1": 514.25}],
            [{"position_cm-1": 514.25, "label": "A1g + B1g"}],
            load_card("anatase_simplified.json"),
            anatase_context,
        )
        _expect(failures, real_peak, True, "real anatase 514 combination")
    mos2_path = MODE_CARDS_DIR / "mos2.json"
    if mos2_path.is_file():
        real_mos2 = card_rules(load_card("mos2.json"))
        if "E'" not in real_mos2.allowed:
            failures.append("real mos2 card missing E'")
        if "Eg" in real_mos2.allowed:
            failures.append("real mos2 card unexpectedly allows Eg")
        if "B1u" not in real_mos2.silent:
            failures.append("real mos2 card missing silent B1u")
    quartz_path = MODE_CARDS_DIR / "quartz.json"
    if quartz_path.is_file():
        real_quartz = card_rules(load_card("quartz.json"))
        if "A2" not in real_quartz.silent:
            failures.append("real quartz card missing silent A2")
        if "A1" not in real_quartz.allowed:
            failures.append("real quartz card missing A1")

    prompt = retry_prompt("CARD\n\nPeak list:\n[]\n", ["peak at 1 cm^-1 has no label"])
    if "answers" in prompt or "gold" in prompt.casefold():
        failures.append("retry prompt mentions gold")
    if "peak at 1 cm^-1 has no label" not in prompt:
        failures.append("retry prompt dropped the reason")

    if failures:
        print("self-test failed", file=sys.stderr)
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    print("self-test passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checker self-test for mode assignments.")
    parser.add_argument("--self-test", action="store_true", help="Run the checker cases and exit.")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    parser.error("pass --self-test. The scoring path is python -m scripts.run_checker.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
