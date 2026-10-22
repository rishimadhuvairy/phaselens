# scripts/spectrum_generator and scripts/peak_finder are extensionless.
# This file uses a .py suffix so it can be run as a module (python -m scripts.<name>).
"""Settings for the PhaseLens zero-shot baseline. No secrets and no API keys.

``ZERO_SHOT_MODEL_ID`` is the official post-trained id ``Qwen/Qwen3.8-27B``.
The launcher still refuses to call Modal when the id is empty or contains
``PLACEHOLDER``.

The informal names "Qwen 3.8-28b" and "Qwen 3.6-25b-a3b" were checked against
Hugging Face and the Qwen release notes on 2026-10-22. Neither name is exactly
one official model id. ``Qwen/Qwen3.8-27B`` is the selected official id.

Closest official post-trained (instruct/chat) ids that were verified:

- ``Qwen/Qwen3.6-35B-A3B`` — Qwen3.6 MoE, 35B total / 3B active. Closest official
  model to "Qwen 3.6-25b-a3b". It is not a 25B model.
- ``Qwen/Qwen3.8-27B`` — Qwen3.8 dense post-trained model. The model card says
  27B parameters. The Hub collection tags the repo as 28B. There is no
  ``Qwen/Qwen3.8-28B`` repo.
- ``Qwen/Qwen3.6-27B`` — Qwen3.6 dense post-trained model. The model card says
  27B parameters. The Hub collection tags the repo as 28B.

These repos are the post-trained chat models. A separate ``-Instruct`` repo was
not found for them. Official FP8 twins exist (``Qwen/Qwen3.6-35B-A3B-FP8``,
``Qwen/Qwen3.8-27B-FP8``, ``Qwen/Qwen3.6-27B-FP8``) and are the same models in
another dtype.

``logic65/Qwen3.6-Whittle-25B-A3B`` is a third-party prune of
``Qwen/Qwen3.6-35B-A3B``. It is not an official Qwen id and is not used here.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Any id containing PLACEHOLDER is refused by the launcher.
PLACEHOLDER_MODEL_ID = "PLACEHOLDER_UNVERIFIED_QWEN_ID"
ZERO_SHOT_MODEL_ID = "Qwen/Qwen3.8-27B"

CLOSEST_OFFICIAL_MODEL_IDS = (
    {
        "id": "Qwen/Qwen3.6-35B-A3B",
        "why": (
            "Official Qwen3.6 post-trained MoE, 35B total and 3B active. "
            "Closest official id to 'Qwen 3.6-25b-a3b'. Not a 25B model."
        ),
    },
    {
        "id": "Qwen/Qwen3.8-27B",
        "why": (
            "Official Qwen3.8 post-trained dense model. The card says 27B "
            "parameters; the Hub collection tags the repo 28B. No Qwen/Qwen3.8-28B repo was found."
        ),
    },
    {
        "id": "Qwen/Qwen3.6-27B",
        "why": (
            "Official Qwen3.6 post-trained dense model. The card says 27B "
            "parameters; the Hub collection tags the repo 28B."
        ),
    },
)

NOT_OFFICIAL_NEARBY = (
    "logic65/Qwen3.6-Whittle-25B-A3B is a third-party prune of "
    "Qwen/Qwen3.6-35B-A3B (about 25B total, 3B active). It is not an official "
    "Qwen model id and is not selected."
)

# Passed through to Modal. modal 1.6.2 only mentions this string as an example
# gpu_type on experimental cluster reservations; it is not renamed here.
GPU = "B300"

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.0
# One remote call carries this many prompts. Inside the call, prompts are
# generated one at a time because each prompt includes a full mode card.
BATCH_SIZE = 4
# One container, so a launch cannot fan out across more than one B300.
MAX_CONTAINERS = 1
# Per-input limit. The first call also downloads weights, so this is longer
# than a warm batch of short JSON answers.
TIMEOUT_S = 7200
STARTUP_TIMEOUT_S = 1800
MEMORY_MIB = 131072
SCALEDOWN_WINDOW_S = 300
APP_NAME = "phaselens-qwen-zeroshot"

HELD_OUT_LASER_NM = 632.8
HELD_OUT_MATERIAL = "calcite"
HELD_OUT_MODE_CARD = "calcite_simplified.json"
LASER_MATCH_ABS_TOL = 1e-6

MODE_SOURCE = "mode"
ARTIFACT_SOURCES = ("plasma", "substrate")
IGNORED_SOURCE = "false_peak"
IGNORED_LABEL = "unassigned"
ARTIFACT_LABEL = "artifact"

TRAINING_DATA_DIR = REPO_ROOT / "training_data"
MODE_CARDS_DIR = REPO_ROOT / "mode_cards"
TRAIN_DIR = REPO_ROOT / "data" / "train"
TEST_DIR = REPO_ROOT / "data" / "test"
SPLIT_MANIFEST = REPO_ROOT / "data" / "split_manifest.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PROMPTS_JSONL = OUTPUTS_DIR / "step5_prompts.jsonl"
PREDICTIONS_JSONL = OUTPUTS_DIR / "step5_predictions.jsonl"
SCORES_JSON = OUTPUTS_DIR / "step5_scores.json"


def model_id_is_placeholder(model_id: str | None = None) -> bool:
    mid = ZERO_SHOT_MODEL_ID if model_id is None else model_id
    if not isinstance(mid, str):
        return True
    token = mid.strip()
    if not token:
        return True
    if token == PLACEHOLDER_MODEL_ID:
        return True
    return "PLACEHOLDER" in token.upper()


def placeholder_refusal_message() -> str:
    lines = [
        "Refusing to launch zero-shot inference.",
        f"ZERO_SHOT_MODEL_ID is {ZERO_SHOT_MODEL_ID!r}, which is still a placeholder.",
        "The informal names 'Qwen 3.8-28b' and 'Qwen 3.6-25b-a3b' do not each match exactly one official Hugging Face id.",
        "Closest official post-trained (instruct/chat) ids:",
    ]
    for item in CLOSEST_OFFICIAL_MODEL_IDS:
        lines.append(f"  - {item['id']}: {item['why']}")
    lines.append(NOT_OFFICIAL_NEARBY)
    lines.append(
        "Edit scripts/qwen_config.py and set ZERO_SHOT_MODEL_ID to a verified id. "
        "This process did not call Modal."
    )
    return "\n".join(lines)


def laser_matches(value: object, target: float = HELD_OUT_LASER_NM) -> bool:
    try:
        return abs(float(value) - float(target)) <= LASER_MATCH_ABS_TOL
    except (TypeError, ValueError):
        return False


def is_held_out_material(material: object = None, mode_card: object = None) -> bool:
    parts: list[str] = []
    for value in (material, mode_card):
        if isinstance(value, str):
            parts.append(value)
    haystack = " ".join(parts).casefold()
    return HELD_OUT_MATERIAL.casefold() in haystack


def is_test_record(material: object = None, mode_card: object = None, laser_nm: object = None) -> bool:
    """Test is the held-out material UNION the held-out laser."""
    return is_held_out_material(material, mode_card) or laser_matches(laser_nm)
