"""PhaseLens demo core: spectrum -> peaks -> labels -> readout.

Reuses the project code vendored in ``scripts/``:
- ``scripts/peak_finder``        (pybaselines arPLS + scipy find_peaks)
- ``scripts/spectrum_generator`` (MOS2_LAYER_TABLE, MOS2_LABELS, PLASMA_LINES)
- ``scripts/checker.py``         (card_rules, 4.0 cm^-1 window)

The assignment on this page is the checker's rules applied directly
(nearest card mode inside 4 cm^-1; plasma / Si lines -> "artifact").
The Qwen model is not run on the free CPU Space.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
from pybaselines import Baseline

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
CARDS_DIR = HERE / "mode_cards"
EXAMPLES_DIR = HERE / "examples"
MOTION_DIR = HERE / "motion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))


def _load_extensionless(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


peak_finder = _load_extensionless("peak_finder", SCRIPTS / "peak_finder")
generator = _load_extensionless("spectrum_generator", SCRIPTS / "spectrum_generator")
from scripts.checker import FREQUENCY_TOLERANCE_CM, card_rules  # noqa: E402

TOL = FREQUENCY_TOLERANCE_CM  # 4.0 cm^-1
LAYER_ORDER = ("1L", "2L", "3L", "4L", "bulk")

MATERIALS = {  # display name -> mode card file
    "MoS₂": "mos2.json",
    "Quartz (α-SiO₂)": "quartz.json",
    "Calcite (CaCO₃)": "calcite_simplified.json",
    "Anatase (TiO₂)": "anatase_simplified.json",
    "Silicon": "silicon.json",
}
LASERS = [488.0, 514.5, 532.0, 632.8]


# ---------------------------------------------------------------- loading
def load_card(name: str) -> dict:
    return json.loads((CARDS_DIR / name).read_text(encoding="utf-8"))


def silicon_line() -> float:
    card = load_card("silicon.json")
    for m in card.get("curated_expected_modes", []):
        if m.get("position_cm-1"):
            return float(m["position_cm-1"])
    return 520.7


def parse_text_spectrum(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Two numeric columns (shift, intensity); commas, tabs or spaces; header lines skipped."""
    rows = []
    for line in io.StringIO(raw.decode("utf-8", errors="ignore")):
        parts = line.replace(",", " ").replace(";", " ").split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 20:
        raise ValueError("Need at least 20 rows of two numeric columns: Raman shift (cm⁻¹), intensity.")
    data = np.array(sorted(rows))
    return data[:, 0], data[:, 1]


def list_examples() -> dict:
    out = {}
    for p in sorted(EXAMPLES_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["title"]] = p.name
    return out


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text())


# ---------------------------------------------------------------- artifacts
def plasma_lines_for(card: dict, laser_nm: float) -> tuple[list[float], str]:
    lines, sources = [], []
    table = generator.PLASMA_LINES.get(laser_nm)
    if table:
        lines += list(table)
        sources.append("Ne lines for the 632.8 nm HeNe laser (scripts/spectrum_generator, NEON_LINES_NM)")
    for item in (card.get("artifact_lines") or {}).get("laser_dependent_plasma_or_stray_light", []) or []:
        lw, pos = item.get("laser_wavelength_nm"), item.get("position_cm-1")
        if lw and pos is not None and abs(float(lw) - laser_nm) < 1.0:
            lines.append(float(pos))
            if "mode card" not in " ".join(sources):
                sources.append("laser lines listed on the mode card (NIST ASD)")
    note = "; ".join(sources) if sources else f"no plasma-line list is stored for {laser_nm:g} nm"
    return sorted(set(round(x, 1) for x in lines)), note


# ---------------------------------------------------------------- peaks
def find_peaks_with_baseline(x: np.ndarray, y: np.ndarray) -> dict:
    baseline, _ = Baseline(x_data=x).arpls(y, lam=peak_finder.BASELINE_LAM)
    peaks = peak_finder.find_raman_peaks(x, y)["peaks"]
    return {"baseline": baseline, "corrected": y - baseline, "peaks": peaks}


# ---------------------------------------------------------------- assignment
def _mos2_layer(e_pos, a_pos, laser):
    by_laser = generator.MOS2_LAYER_TABLE.get(laser)
    table_laser = laser
    if by_laser is None:
        table_laser = min(generator.MOS2_LAYER_TABLE, key=lambda k: abs(k - laser))
        if abs(table_laser - laser) > 1.0:
            table_laser = 532.0
        by_laser = generator.MOS2_LAYER_TABLE[table_laser]

    def key(layer):
        e_ref, a_ref = by_laser[layer]
        return (abs(e_pos - e_ref) + abs(a_pos - a_ref), abs((a_pos - e_pos) - (a_ref - e_ref)))

    layer = min(LAYER_ORDER, key=key)
    return layer, by_laser, table_laser


def assign(peaks: list[dict], card_file: str, laser_nm: float, on_silicon: bool,
           plasma: list[float], si_line: float) -> tuple[list[dict], dict]:
    card = load_card(card_file)
    is_si_sample = card_file.startswith("silicon")
    is_mos2 = card_file.startswith("mos2")
    rules = card_rules(card)
    rows = []
    for p in peaks:
        x = float(p["position_cm-1"])
        row = {"position_cm-1": x, "height": p["height"], "fwhm_cm-1": p["fwhm_cm-1"],
               "label": "unassigned", "reason": "no card mode within 4 cm⁻¹", "card_cm-1": None}
        near_plasma = [q for q in plasma if abs(q - x) <= TOL]
        if near_plasma:
            row.update(label="artifact", reason=f"laser plasma line at {near_plasma[0]:.1f} cm⁻¹")
            overlap = [m for m in rules.modes if abs(m.position - x) <= TOL] if not is_mos2 else []
            if overlap:
                row["reason"] += (f" — note: also within 4 cm⁻¹ of card mode {overlap[0].name} "
                                  f"({overlap[0].position:g}); the checker rule labels it artifact")
        elif on_silicon and not is_si_sample and abs(x - si_line) <= TOL:
            row.update(label="artifact", reason=f"Si substrate line ({si_line:.1f} cm⁻¹)")
        elif not is_mos2:
            near = sorted((m for m in rules.modes if abs(m.position - x) <= TOL),
                          key=lambda m: abs(m.position - x))
            if near:
                row.update(label=near[0].name,
                           reason=f"card mode at {near[0].position:g} cm⁻¹ (Δ = {x - near[0].position:+.1f})")
                row["card_cm-1"] = near[0].position
        rows.append(row)

    readout = {"kind": None, "sentence": None}
    if is_mos2:
        by_laser = generator.MOS2_LAYER_TABLE.get(laser_nm) or generator.MOS2_LAYER_TABLE[532.0]
        e_refs = [v[0] for v in by_laser.values()]
        a_refs = [v[1] for v in by_laser.values()]
        free = [r for r in rows if r["label"] == "unassigned"]
        e_c = [r for r in free if min(e_refs) - TOL <= r["position_cm-1"] <= max(e_refs) + TOL]
        a_c = [r for r in free if min(a_refs) - TOL <= r["position_cm-1"] <= max(a_refs) + TOL]
        e_row = max(e_c, key=lambda r: r["height"]) if e_c else None
        a_row = max(a_c, key=lambda r: r["height"]) if a_c else None
        if e_row and a_row:
            layer, table, table_laser = _mos2_layer(e_row["position_cm-1"], a_row["position_cm-1"], laser_nm)
            group = "bulk" if layer == "bulk" else ("odd" if layer in ("1L", "3L") else "even")
            e_lab, a_lab = generator.MOS2_LABELS[group]
            gap = a_row["position_cm-1"] - e_row["position_cm-1"]
            e_ref, a_ref = table[layer]
            e_row.update(label=e_lab, reason=f"in-plane MoS₂ mode; {layer} reference {e_ref:g} cm⁻¹")
            a_row.update(label=a_lab, reason=f"out-of-plane MoS₂ mode; {layer} reference {a_ref:g} cm⁻¹")
            e_row["card_cm-1"], a_row["card_cm-1"] = e_ref, a_ref
            layer_txt = "bulk" if layer == "bulk" else f"{layer[0]}-layer"
            readout = {
                "kind": "layer", "layer": layer, "gap": gap,
                "sentence": (f"Layer readout: the out-of-plane peak ({a_row['position_cm-1']:.1f}) minus the in-plane peak "
                             f"({e_row['position_cm-1']:.1f}) is **{gap:.1f} cm⁻¹**. The closest row of the layer table "
                             f"for {table_laser:g} nm is **{layer_txt} MoS₂** (reference {e_ref:g} / {a_ref:g} cm⁻¹, "
                             f"gap {a_ref - e_ref:.1f} cm⁻¹)."),
                "table": table, "table_laser": table_laser,
            }
        else:
            readout = {"kind": "layer", "sentence": "Layer readout: an in-plane and an out-of-plane MoS₂ peak were not both found, so no layer number is given."}
        readout["strain"] = ("Strain: not computed. The MoS₂ mode card has no numeric strain slope, "
                             "and this page does not invent one.")
    else:
        assigned = [r for r in rows if r.get("card_cm-1") is not None and r["label"] not in ("artifact", "unassigned")]
        if assigned:
            shifts = [r["position_cm-1"] - r["card_cm-1"] for r in assigned]
            mean = float(np.mean(shifts))
            same = all(s > 0.5 for s in shifts) or all(s < -0.5 for s in shifts)
            if same and len(shifts) >= 2:
                direction = "up" if mean > 0 else "down"
                txt = (f"Strain check: all {len(shifts)} assigned peaks moved {direction} from the card positions "
                       f"(mean {mean:+.1f} cm⁻¹). That pattern is consistent with strain (or temperature / calibration), "
                       "but the card has no slope, so no strain value is given.")
            elif len(shifts) == 1:
                txt = (f"Strain check: one assigned peak, {shifts[0]:+.1f} cm⁻¹ from the card position. "
                       "The card has no strain slope, so no strain value is given.")
            else:
                txt = (f"Strain check: assigned peaks did not all move the same way from the card positions "
                       f"(mean shift {mean:+.1f} cm⁻¹), so no strain is reported.")
            readout = {"kind": "strain", "sentence": txt}
        else:
            readout = {"kind": "strain", "sentence": "Strain check: no peak was assigned to a card mode."}
    return rows, readout


# ---------------------------------------------------------------- motion pictures
def motion_picture(card_file: str, label: str, x: float) -> tuple[Path | None, str]:
    lab = label
    if card_file.startswith("mos2"):
        if lab in ("E'", "Eg", "E1_2g"):
            return MOTION_DIR / "mos2_inplane.png", "in-plane"
        if lab in ("A1'", "A1g"):
            return MOTION_DIR / "mos2_outplane.png", "out-of-plane"
    if card_file.startswith("silicon") and lab == "F2g":
        return MOTION_DIR / "si_f2g.png", ""
    if card_file.startswith("calcite"):
        if lab == "A1g":
            return MOTION_DIR / "calcite_nu1.png", ""
        if lab.startswith("overtone"):
            return MOTION_DIR / "calcite_2nu2.png", ""
        if lab == "Eg":
            name = ("calcite_T.png" if x < 210 else "calcite_L.png" if x < 450
                    else "calcite_nu4.png" if x < 900 else "calcite_nu3.png")
            return MOTION_DIR / name, ""
    if card_file.startswith("anatase"):
        if lab == "Eg":
            name = "anatase_eg_external.png" if x < 170 else "anatase_eg_bend.png" if x < 300 else "anatase_eg_stretch.png"
            return MOTION_DIR / name, ""
        if lab == "B1g" and x < 450:
            return MOTION_DIR / "anatase_b1g.png", ""
        if lab in ("A1g + B1g", "A1g", "B1g"):
            return MOTION_DIR / "anatase_a1g_b1g.png", ""
    if card_file.startswith("quartz"):
        return None, "The quartz mode card has no motion text (its `motion` fields are null), so no picture is shown."
    return None, "No saved picture for this label."


# ---------------------------------------------------------------- citations
def citations(card_file: str) -> list[str]:
    card = load_card(card_file)
    out = []
    cit = card.get("citation")
    if isinstance(cit, dict):
        out.append(cit.get("primary", ""))
        if cit.get("experimental_source_cited_in_paper"):
            out.append("Experimental values cited in that paper: " + cit["experimental_source_cited_in_paper"])
    for s in card.get("studies", []) or []:
        if s.get("source_verified") is False:
            continue
        text = s.get("citation", "")
        doi = s.get("doi")
        url = None
        if not doi:
            for m in card.get("curated_expected_modes", []):
                prov = m.get("provenance") or {}
                if prov.get("url"):
                    url = prov["url"]
                    break
        if doi:
            text += f" — https://doi.org/{doi}"
        elif url:
            text += f" — {url}"
        out.append(text)
        if card_file.startswith(("anatase", "calcite")):
            break
    return [c for c in out if c]


def run(x, y, card_file, laser_nm, on_silicon, plasma=None, si_line=None):
    card = load_card(card_file)
    if plasma is None:
        plasma, plasma_note = plasma_lines_for(card, laser_nm)
    else:
        plasma_note = "artifact lines stored with this example"
    si = si_line if si_line is not None else silicon_line()
    pk = find_peaks_with_baseline(np.asarray(x, float), np.asarray(y, float))
    rows, readout = assign(pk["peaks"], card_file, laser_nm, on_silicon, plasma, si)
    return {"baseline": pk["baseline"], "corrected": pk["corrected"], "rows": rows, "readout": readout,
            "plasma": plasma, "plasma_note": plasma_note, "si_line": si if on_silicon else None,
            "citations": citations(card_file)}
