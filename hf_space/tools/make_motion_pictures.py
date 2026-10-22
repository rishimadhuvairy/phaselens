"""Draw the saved atomic-motion pictures used by the PhaseLens page.

Each picture is a 2-D schematic: circles are atoms, arrows are displacements.
The motion drawn for each mode follows the motion text on the mode card
(anatase, calcite) or the standard textbook assignment (MoS2, Si).
Quartz has no motion text on its card, so no quartz picture is drawn.

Run:  python tools/make_motion_pictures.py   (writes motion/*.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "motion"
OUT.mkdir(exist_ok=True)

COLORS = {"Mo": "#7b5ea7", "S": "#e1b12c", "Si": "#5a7d9a", "Ti": "#9aa5b1",
          "O": "#d64541", "C": "#333333", "Ca": "#3fa34d"}
SIZES = {"Mo": 0.30, "S": 0.22, "Si": 0.26, "Ti": 0.28, "O": 0.18, "C": 0.16, "Ca": 0.30}


def new(title, subtitle, w=5.2, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    fig.text(0.5, 0.025, subtitle, ha="center", fontsize=8.5, color="#444", wrap=True)
    return fig, ax


def atom(ax, x, y, el, label=True):
    ax.add_patch(Circle((x, y), SIZES[el], fc=COLORS[el], ec="black", lw=0.8, zorder=3))
    if label:
        ax.text(x, y, el, ha="center", va="center", fontsize=7,
                color="white" if el in ("Mo", "C", "Si") else "black", zorder=4)


def bond(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", lw=1.4, zorder=1)


def arrow(ax, x, y, dx, dy, color="#0b6efd"):
    ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>", mutation_scale=13,
                                 lw=2.0, color=color, zorder=5))


def save(fig, name, xlim, ylim):
    ax = fig.axes[0]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- MoS2 (side view of one S-Mo-S layer) ----------------
def mos2_layer(ax, mode):
    xs = [0, 1.6, 3.2]
    for x in xs:
        s_top, mo, s_bot = (x + 0.8, 1.0), (x, 0.0), (x + 0.8, -1.0)
        bond(ax, mo, s_top); bond(ax, mo, s_bot)
        if x < 3.2:
            bond(ax, (x + 1.6, 0.0), s_top); bond(ax, (x + 1.6, 0.0), s_bot)
    for x in xs:
        atom(ax, x, 0.0, "Mo")
        atom(ax, x + 0.8, 1.0, "S"); atom(ax, x + 0.8, -1.0, "S")
        if mode == "inplane":
            arrow(ax, x, 0.0, 0.55, 0)               # Mo one way
            arrow(ax, x + 0.8, 1.0, -0.45, 0)         # both S the other way
            arrow(ax, x + 0.8, -1.0, -0.45, 0)
        else:
            arrow(ax, x + 0.8, 1.0, 0, 0.55)          # top S up
            arrow(ax, x + 0.8, -1.0, 0, -0.55)        # bottom S down; Mo still
    ax.annotate("c axis", xy=(-0.9, 1.0), xytext=(-0.9, -0.6), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#555"))


fig, ax = new("MoS₂ in-plane mode  (E′ / E¹₂g / Eg)",
              "Mo moves one way in the layer plane; both S planes move the opposite way. "
              "Side view of one S–Mo–S layer; in bulk the two layers of the cell move out of phase.")
mos2_layer(ax, "inplane")
save(fig, "mos2_inplane.png", (-1.4, 4.8), (-1.8, 1.8))

fig, ax = new("MoS₂ out-of-plane mode  (A₁′ / A₁g)",
              "The two S planes move apart along c, in opposite directions; Mo stays still. "
              "Side view of one S–Mo–S layer.")
mos2_layer(ax, "outplane")
save(fig, "mos2_outplane.png", (-1.4, 4.8), (-1.8, 1.9))

# ---------------- Si (diamond; two fcc sublattices) ----------------
fig, ax = new("Si optical phonon  (F₂g, ≈520 cm⁻¹)",
              "The two interpenetrating fcc sublattices move rigidly against each other "
              "(zone-centre optical phonon). 2-D projection of the bonded chain.")
pts = [(i * 1.0, 0.6 if i % 2 else -0.6) for i in range(6)]
for a, b in zip(pts, pts[1:]):
    bond(ax, a, b)
for i, (x, y) in enumerate(pts):
    atom(ax, x, y, "Si")
    arrow(ax, x, y, 0.32 if i % 2 else -0.32, 0.32 if i % 2 else -0.32,
          color="#0b6efd" if i % 2 else "#e8590c")
ax.text(5.6, 0.95, "sublattice B", color="#0b6efd", fontsize=8)
ax.text(5.6, -1.15, "sublattice A", color="#e8590c", fontsize=8)
save(fig, "si_f2g.png", (-0.8, 7.0), (-1.6, 1.6))


# ---------------- Calcite: CO3 in the plane ----------------
import math


def co3(ax, cx=0.0, cy=0.0, r=1.0):
    pos = [(cx + r * math.cos(math.radians(90 + 120 * k)), cy + r * math.sin(math.radians(90 + 120 * k)))
           for k in range(3)]
    for p in pos:
        bond(ax, (cx, cy), p)
    atom(ax, cx, cy, "C")
    for p in pos:
        atom(ax, *p, "O")
    return pos


fig, ax = new("Calcite ν₁  (A₁g, ≈1085 cm⁻¹)",
              "Symmetric C–O stretch: all three O move radially out (then in) together; C stays still.")
for (x, y) in co3(ax):
    arrow(ax, x, y, 0.45 * x, 0.45 * y)
save(fig, "calcite_nu1.png", (-1.8, 1.8), (-1.5, 1.8))

fig, ax = new("Calcite ν₃  (Eg, ≈1434 cm⁻¹)",
              "Antisymmetric C–O stretch: one C–O bond lengthens while the other two shorten; "
              "C moves to keep the centre of mass fixed. One of the two degenerate partners shown.")
o = co3(ax)
arrow(ax, *o[0], 0, 0.45)
arrow(ax, *o[1], 0.35 * (-o[1][0]), 0.35 * (-o[1][1]))
arrow(ax, *o[2], 0.35 * (-o[2][0]), 0.35 * (-o[2][1]))
arrow(ax, 0, 0, 0, -0.35)
save(fig, "calcite_nu3.png", (-1.8, 1.8), (-1.5, 1.9))

fig, ax = new("Calcite ν₄  (Eg, ≈711 cm⁻¹)",
              "In-plane O–C–O bend: two O atoms swing toward each other around C, "
              "changing the O–C–O angle. One degenerate partner shown.")
o = co3(ax)
for p, s in ((o[1], 1), (o[2], -1)):
    tx, ty = -p[1], p[0]          # tangent
    arrow(ax, *p, 0.4 * s * tx, 0.4 * s * ty)
save(fig, "calcite_nu4.png", (-1.8, 1.8), (-1.6, 1.6))


def lattice(ax, mode):
    for x in (-2.4, 2.4):
        atom(ax, x, 0, "Ca")
        if mode == "T":
            arrow(ax, x, 0, 0.5, 0, color="#e8590c")
    o = co3(ax, 0, 0, 0.9)
    if mode == "T":
        for p in o + [(0, 0)]:
            arrow(ax, *p, -0.5, 0)
    else:
        arc = FancyArrowPatch((0.9, 1.0), (-0.9, 1.0), connectionstyle="arc3,rad=0.5",
                              arrowstyle="-|>", mutation_scale=14, lw=2, color="#0b6efd")
        ax.add_patch(arc)
        arc2 = FancyArrowPatch((-0.9, -1.0), (0.9, -1.0), connectionstyle="arc3,rad=0.5",
                               arrowstyle="-|>", mutation_scale=14, lw=2, color="#0b6efd")
        ax.add_patch(arc2)


fig, ax = new("Calcite lattice mode T  (Eg, ≈154 cm⁻¹)",
              "External mode: the whole CO₃ group translates as a rigid unit against the Ca ions.")
lattice(ax, "T")
save(fig, "calcite_T.png", (-3.0, 3.0), (-1.4, 1.4))

fig, ax = new("Calcite lattice mode L  (Eg, ≈281 cm⁻¹)",
              "External mode: libration (rocking) of the rigid CO₃ group between the Ca ions.")
lattice(ax, "L")
save(fig, "calcite_L.png", (-3.0, 3.0), (-1.9, 1.9))

fig, ax = new("Calcite 2ν₂ overtone  (≈1748 cm⁻¹)",
              "Second-order band: twice the out-of-plane bend ν₂, where C moves along c "
              "opposite to the three O atoms (⊙ = toward you, ⊗ = away).")
o = co3(ax)
ax.text(0.28, 0.22, "⊙", fontsize=16, color="#0b6efd")
for p in o:
    ax.text(p[0] + 0.2, p[1] + 0.15, "⊗", fontsize=14, color="#0b6efd")
save(fig, "calcite_2nu2.png", (-1.8, 1.8), (-1.5, 1.6))


# ---------------- Anatase: one O-Ti-O unit along c ----------------
def tio(ax):
    ti, o_up, o_dn = (0, 0), (0, 1.3), (0, -1.3)
    o_l, o_r = (-1.3, -0.25), (1.3, 0.25)
    for p in (o_up, o_dn, o_l, o_r):
        bond(ax, ti, p)
    atom(ax, *ti, "Ti")
    for p in (o_up, o_dn, o_l, o_r):
        atom(ax, *p, "O")
    ax.annotate("c", xy=(-2.1, 1.2), xytext=(-2.1, 0.0), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555"))
    return ti, o_up, o_dn, o_l, o_r


def anatase(name, title, sub, moves):
    fig, ax = new(title, sub)
    pts = dict(zip(("Ti", "Oup", "Odn", "Ol", "Or"), tio(ax)))
    for key, (dx, dy) in moves.items():
        arrow(ax, *pts[key], dx, dy)
    save(fig, name, (-2.6, 2.4), (-1.9, 1.9))


anatase("anatase_eg_external.png", "Anatase Eg(1)  (≈145 cm⁻¹)",
        "Card: O–Ti–O bending / external mode; Ti and O move in the a–b plane "
        "(Ti against the O cage). Schematic.",
        {"Ti": (0.45, 0), "Oup": (-0.4, 0), "Odn": (-0.4, 0)})
anatase("anatase_eg_bend.png", "Anatase Eg(2)  (≈199 cm⁻¹)",
        "Card: O–Ti–O bending in the a–b plane; weak band. Schematic.",
        {"Oup": (0.45, 0), "Odn": (-0.45, 0)})
anatase("anatase_b1g.png", "Anatase B1g(1)  (≈396 cm⁻¹)",
        "Card: O–Ti–O symmetric bending, O atoms moving along c in opposite directions. Schematic.",
        {"Ol": (0, 0.45), "Or": (0, -0.45)})
anatase("anatase_a1g_b1g.png", "Anatase A1g + B1g(2)  (≈517 cm⁻¹)",
        "Card: two nearly degenerate O–Ti–O antisymmetric bending modes along c, "
        "unresolved at room temperature. Schematic.",
        {"Ol": (0, 0.45), "Or": (0, 0.45), "Ti": (0, -0.35)})
anatase("anatase_eg_stretch.png", "Anatase Eg(3)  (≈638 cm⁻¹)",
        "Card: Ti–O symmetric stretching in the a–b plane. Schematic.",
        {"Ol": (-0.45, -0.08), "Or": (0.45, 0.08)})

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
