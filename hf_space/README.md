---
title: PhaseLens
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Raman peaks to symmetry names, artifacts, MoS2 layers
---

# PhaseLens

Load a Raman spectrum or pick a built-in example. The page shows the raw curve, the removed
background, the peak list, the symmetry name for each peak, a saved picture of the atomic motion,
"artifact" on plasma and silicon lines, the layer or strain sentence, and the mode-card citation.

Code: https://github.com/rishimadhuvairy/phaselens

Files: `app.py` (marimo page), `phaselens_core.py` (pipeline), `scripts/` (peak finder, spectrum
generator, checker from the repo), `mode_cards/`, `examples/` (synthetic test-split spectra),
`motion/` (saved motion pictures, drawn by `tools/make_motion_pictures.py`).

## This folder (`hf_space/` in the GitHub repo)

This is the exact content of the Hugging Face Space
https://huggingface.co/spaces/yqm1221/phaselens . It is self-contained, so it does not
touch the repo-root `app.py` / `backend_adapter.py`.

- `scripts/` is a copy of the repo's `scripts/` files the page needs (`peak_finder`,
  `spectrum_generator`, `checker.py`, `label_norm.py`, `qwen_config.py`, `__init__.py`),
  copied from repo commit 573d580. If those change in the repo, copy them here again.
- `mode_cards/` is a copy of the repo's cards (simplified anatase/calcite are the ones the test files name).
- `examples/` are six synthetic spectra from `data/test` (632.8 nm), with the generator's known answers.
- Labels on the page use the checker's rules directly (nearest card mode within 4 cm^-1; plasma and
  Si lines -> "artifact"). Qwen is not run on the free CPU Space.

Run locally:

```bash
cd hf_space
pip install -r requirements.txt
marimo run app.py        # or: marimo edit app.py
```

Deploy: upload the contents of this folder to the Space root (the template `Dockerfile` runs
`marimo run app.py` on port 7860).
