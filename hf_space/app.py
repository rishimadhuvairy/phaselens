import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="PhaseLens")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phaselens_core as pc
    return mo, np, pc, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # PhaseLens — Raman peak → symmetry name

        Load a Raman spectrum or pick a built-in example. PhaseLens removes the fluorescence background,
        finds the peaks, gives each peak a **symmetry name** from the material's mode card (or **artifact**
        for laser-plasma and silicon-substrate lines), shows how the atoms move for that mode,
        and reads the **MoS₂ layer number** or a **strain check**.
        """
    )
    return


@app.cell
def _(mo, pc):
    source = mo.ui.radio(options=["Built-in example", "Upload my spectrum"],
                         value="Built-in example", label="**Spectrum source**", inline=True)
    _examples = pc.list_examples()
    example = mo.ui.dropdown(options=_examples, value=list(_examples)[0], label="Example")
    upload = mo.ui.file(filetypes=[".txt", ".csv", ".dat", ".xy"], kind="area",
                        label="Two columns: Raman shift (cm⁻¹), intensity")
    material = mo.ui.dropdown(options=pc.MATERIALS, value="MoS₂", label="Material (mode card)")
    laser = mo.ui.dropdown(options={f"{l:g} nm": l for l in pc.LASERS}, value="532 nm", label="Laser")
    on_si = mo.ui.checkbox(value=True, label="Sample sits on a Si / SiO₂ substrate")
    return example, laser, material, on_si, source, upload


@app.cell
def _(example, laser, material, mo, on_si, source, upload):
    if source.value == "Built-in example":
        _controls = mo.vstack([source, example])
    else:
        _controls = mo.vstack([source, upload, mo.hstack([material, laser, on_si], justify="start")])
    _controls
    return


@app.cell
def _(example, laser, material, mo, on_si, pc, source, upload):
    if source.value == "Built-in example":
        ex = pc.load_example(example.value)
        spec = dict(x=ex["shift_cm-1"], y=ex["intensity"], card=ex["mode_card"], laser=ex["laser_nm"],
                    on_si=ex["on_silicon"], plasma=ex["plasma_cm-1"], si=ex["si_line"],
                    note=ex["description"], answers=ex["answers"], true_layer=ex["true_layer"],
                    origin=f"Synthetic spectrum from the project's spectrum generator (test split, `{ex['source_file']}`).")
    else:
        mo.stop(not upload.value, mo.callout(mo.md("Upload a two-column text file to start."), kind="info"))
        try:
            _x, _y = pc.parse_text_spectrum(upload.contents())
        except ValueError as _e:
            mo.stop(True, mo.callout(mo.md(str(_e)), kind="danger"))
        ex = None
        spec = dict(x=_x, y=_y, card=material.value, laser=laser.value, on_si=on_si.value,
                    plasma=None, si=None, note="", answers=None, true_layer=None,
                    origin=f"Uploaded file `{upload.name()}`.")
    result = pc.run(spec["x"], spec["y"], spec["card"], spec["laser"], spec["on_si"],
                    plasma=spec["plasma"], si_line=spec["si"])
    mo.md(f"**{spec['origin']}** {spec['note']}  \nMode card: `{spec['card']}` · laser {spec['laser']:g} nm · "
          f"Si substrate: {'yes' if spec['on_si'] else 'no'} · plasma lines: {result['plasma_note']}")
    return result, spec


@app.cell
def _(np, plt, result, spec):
    _x = np.asarray(spec["x"], float)
    _y = np.asarray(spec["y"], float)
    fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.25]})
    _ax1.plot(_x, _y, color="#222", lw=1.0, label="raw spectrum")
    _ax1.plot(_x, result["baseline"], color="#e8590c", lw=1.6, ls="--", label="removed background (arPLS)")
    _ax1.fill_between(_x, result["baseline"], _y.min(), color="#e8590c", alpha=0.08)
    _ax1.set_ylabel("intensity (a.u.)")
    _ax1.legend(loc="upper right", fontsize=9, frameon=False)
    _ax1.set_title("1 · Raw curve and the background that is removed", loc="left", fontsize=11)

    _ax2.plot(_x, result["corrected"], color="#1c7ed6", lw=1.0)
    _top = max(result["corrected"].max(), 1e-9)
    _prev, _lvl = -1e9, 0
    for _r in sorted(result["rows"], key=lambda r: r["position_cm-1"]):
        _lvl = (_lvl + 1) % 2 if _r["position_cm-1"] - _prev < 40 else 0
        _prev = _r["position_cm-1"]
        _art = _r["label"] == "artifact"
        _col = "#d6336c" if _art else ("#868e96" if _r["label"] == "unassigned" else "#2b8a3e")
        _ax2.axvline(_r["position_cm-1"], color=_col, lw=0.8, ls=":" if _art else "-", alpha=0.7)
        _ax2.annotate(f"{_r['label']}\n{_r['position_cm-1']:.1f}", (_r["position_cm-1"], _r["height"]),
                      xytext=(0, 6 + 26 * _lvl), textcoords="offset points", ha="center", fontsize=8,
                      color=_col, fontweight="bold")
    _ax2.set_ylim(top=_top * 1.5)
    _ax2.set_xlabel("Raman shift (cm⁻¹)")
    _ax2.set_ylabel("background-subtracted")
    _ax2.set_title("2 · Peaks with their labels (green = mode, red dotted = artifact, grey = unassigned)",
                   loc="left", fontsize=11)
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo, result, spec):
    _truth = {}
    if spec["answers"]:
        _truth = {round(a["position_cm-1"], 2): a["label"] for a in spec["answers"]}
    _rows = []
    for _r in result["rows"]:
        _row = {"peak (cm⁻¹)": round(_r["position_cm-1"], 1),
                "FWHM (cm⁻¹)": _r["fwhm_cm-1"],
                "label": _r["label"],
                "why": _r["reason"]}
        if _truth:
            _t = _truth.get(round(_r["position_cm-1"], 2))
            _row["known answer"] = _t if _t is not None else "—"
            _row["match"] = "✓" if _t == _r["label"] else "✗"
        _rows.append(_row)
    _n_ok = sum(1 for _r in _rows if _r.get("match") == "✓")
    _hdr = "### 3 · Peak list and symmetry names"
    if _truth:
        _hdr += f"\nThe generator planted these peaks, so the right answer is known: **{_n_ok}/{len(_rows)}** labels match."
    mo.vstack([mo.md(_hdr), mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell
def _(mo, result, spec):
    _ro = result["readout"]
    _lines = [_ro.get("sentence") or ""]
    if _ro.get("strain"):
        _lines.append(_ro["strain"])
    if spec["true_layer"]:
        _lines.append(f"(Known answer for this synthetic spectrum: **{spec['true_layer']}**.)")
    _arts = [r for r in result["rows"] if r["label"] == "artifact"]
    if _arts:
        _lines.append("Artifacts: " + "; ".join(f"{r['position_cm-1']:.1f} cm⁻¹ — {r['reason']}" for r in _arts) + ".")
    mo.vstack([mo.md("### 4 · Layer or strain readout"),
               mo.callout(mo.md("\n\n".join(_lines)), kind="success" if _ro.get("layer") else "neutral")])
    return


@app.cell
def _(mo, pc, result, spec):
    _seen, _items = set(), []
    for _r in result["rows"]:
        if _r["label"] in ("artifact", "unassigned"):
            continue
        _path, _msg = pc.motion_picture(spec["card"], _r["label"], _r["position_cm-1"])
        _key = str(_path) if _path else _msg
        if _key in _seen:
            continue
        _seen.add(_key)
        _cap = mo.md(f"**{_r['label']}** at {_r['position_cm-1']:.1f} cm⁻¹")
        if _path is not None and _path.exists():
            _items.append(mo.vstack([_cap, mo.image(src=str(_path), width=330)]))
        else:
            _items.append(mo.vstack([_cap, mo.md(f"_{_msg}_")]))
    _body = mo.hstack(_items, wrap=True, justify="start", gap=1) if _items else mo.md("_No mode was assigned._")
    mo.vstack([mo.md("### 5 · How the atoms move (saved pictures)"),
               mo.md("Schematic drawings made for this demo from the motion text on each mode card "
                     "(MoS₂ and Si: standard textbook motions). Arrows show displacement directions, not amplitudes."),
               _body])
    return


@app.cell
def _(mo, pc, result, spec):
    _cits = "\n".join(f"- {c}" for c in result["citations"]) or "- (no citation on this card)"
    _extra = ""
    if spec["card"].startswith("mos2"):
        _extra = ("\n\nLayer table: `MOS2_LAYER_TABLE` in `scripts/spectrum_generator` "
                  "(the project attributes the layer trend to Lee et al. 2010).")
    _si = ("\n\nSilicon line: Parker, Feldman & Ashkin, *Phys. Rev.* **155**, 712 (1967) — "
           "https://doi.org/10.1103/PhysRev.155.712" if spec["on_si"] and not spec["card"].startswith("silicon") else "")
    mo.vstack([mo.md("### 6 · Paper citation from the mode card"), mo.md(_cits + _extra + _si)])
    return


@app.cell
def _(mo):
    mo.accordion({
        "How the labels are made on this page": mo.md(
            """
            - **Background**: pybaselines arPLS (λ = 1e5), as in `scripts/peak_finder`.
            - **Peaks**: `scipy.signal.find_peaks` on the smoothed, background-subtracted curve (SNR ≥ 5).
            - **Labels**: the project checker's rules applied directly — a peak within 4 cm⁻¹ of a laser-plasma
              line is *artifact*; a peak within 4 cm⁻¹ of the Si line on a Si substrate (non-Si sample) is *artifact*;
              otherwise the nearest mode-card frequency within 4 cm⁻¹ gives the symmetry name.
              MoS₂ in-plane/out-of-plane peaks are matched to the layer table, and the label (E′/A₁′, Eg/A₁g, E¹₂g/A₁g)
              follows the layer parity as in the spectrum generator.
            - The fine-tuned Qwen model is **not** run here (a 27B model does not fit on a free CPU Space);
              this page shows the rule-based labels.
            - Plasma lines are known for 632.8 nm (Ne lines) and for any laser lines listed on the mode card.
              For other lasers no plasma-line list is stored, so uploaded spectra at those lasers get no plasma check.
            """
        )
    })
    return


if __name__ == "__main__":
    app.run()
