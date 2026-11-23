"""Shared figure kit for the v3 figures: card layout, header strips, vector icons and statistic chips.

Every figure is a millimetre canvas (overlay axes, drawn first) on which rounded cards carry a coloured
header strip with a panel badge, a short heading and an icon; the data axes sit inside the cards.
Icons are matplotlib primitives (no external images, no generative imagery).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, Arc, FancyArrowPatch)
import iconkit as ik

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#5B6770", "axes.labelcolor": "#26313A",
                     "xtick.color": "#26313A", "ytick.color": "#26313A", "xtick.labelsize": 6.3,
                     "ytick.labelsize": 6.3, "axes.labelsize": 7, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "pdf.fonttype": 42})
INK, GREY, LG = "#26313A", "#7D8A94", "#C9D1D7"
TEAL, TERRA, GOLD, PURP, GREEN, CYAN = "#1F7A8C", "#C0694E", "#B8954A", "#6E6AA6", "#3F7F5F", "#2EC4D6"
TINT = {TEAL: "#E3F1F3", TERRA: "#F8E9E3", GOLD: "#F6EFDF", PURP: "#ECEBF5", GREEN: "#E5F0E9", GREY: "#F1F4F6",
        INK: "#E9ECEE"}
PHYS, SCREEN, AUTH = TERRA, TEAL, GOLD

FIG = None
OV = None
W = H = None


def canvas(w_mm, h_mm):
    global FIG, OV, W, H
    W, H = float(w_mm), float(h_mm)
    FIG = plt.figure(figsize=(W / 25.4, H / 25.4))
    OV = FIG.add_axes([0, 0, 1, 1], zorder=0)
    OV.set_xlim(0, W); OV.set_ylim(0, H); OV.axis("off")
    ik.AX = OV
    return FIG, OV


def save(path_png, path_pdf=None):
    FIG.savefig(path_png, dpi=600, facecolor="white")
    if path_pdf:
        FIG.savefig(path_pdf, facecolor="white")
    plt.close(FIG)


# ------------------------------------------------------------------ primitives (mm coordinates)
def rbox(x, y, w, h, fc="white", ec=GREY, lw=0.6, r=1.6, z=1, ls="-", alpha=1.0):
    OV.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, ls=ls, alpha=alpha))


def t(x, y, s, size=6.0, w="normal", c=INK, ha="center", va="center", z=10, style="normal", rot=0):
    OV.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, zorder=z, fontstyle=style, rotation=rot,
            linespacing=1.15)


def arr(x0, y0, x1, y1, c=GREY, lw=0.7, ms=4, ls="-", rad=0.0, z=6, style="-|>"):
    OV.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 zorder=z, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def chip(x, y, s, c, size=5.6, w=None, fill=True, bold=False):
    """Rounded statistic label centred at (x, y)."""
    w = w or (0.52 * size * len(s) / 3.0 + 3.0)
    rbox(x - w / 2, y - 1.9, w, 3.8, fc=TINT.get(c, "white") if fill else "white", ec=c, lw=0.45, r=1.0, z=12)
    t(x, y, s, size, "bold" if bold else "normal", c if not fill else INK, z=13)


def card(x, y, w, h, letter, title, color, icon=None, pad=(9.0, 7.5, 2.5, 1.8), axes=True):
    """Card with header strip; returns the data axes inside (pad = left, bottom, right, top in mm)."""
    rbox(x, y, w, h, fc="white", ec=color, lw=0.6, r=1.8, z=1)
    OV.add_patch(FancyBboxPatch((x + 0.3, y + h - 6.2), w - 0.6, 5.9, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=TINT[color], ec="none", zorder=2))
    OV.add_patch(Rectangle((x + 0.3, y + h - 6.2), w - 0.6, 2.0, fc=TINT[color], ec="none", zorder=2))
    OV.plot([x + 0.3, x + w - 0.3], [y + h - 6.2, y + h - 6.2], color=color, lw=0.5, zorder=3)
    OV.add_patch(Circle((x + 3.6, y + h - 3.1), 2.1, fc=color, ec="white", lw=0.5, zorder=5))
    t(x + 3.6, y + h - 3.15, letter, 6.6, "bold", "white", z=6)
    t(x + 6.6, y + h - 3.1, title, 6.6, "bold", INK, ha="left")
    if icon is not None:
        icon(x + w - 4.6, y + h - 3.1, 2.2, color)
    if not axes:
        return None
    pl, pb, pr, pt = pad
    a = FIG.add_axes([(x + pl) / W, (y + pb) / H, (w - pl - pr) / W, (h - 6.2 - pb - pt) / H], zorder=5)
    a.set_facecolor("none")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    return a


def legend_item(x, y, kind, c, s, size=5.8):
    if kind == "dot":
        OV.add_patch(Circle((x, y), 0.9, fc=c, ec="white", lw=0.3, zorder=12))
    elif kind == "line":
        OV.plot([x - 1.6, x + 1.6], [y, y], color=c, lw=1.2, zorder=12)
    elif kind == "star":
        OV.scatter([x], [y], marker="*", s=22, color=c, zorder=12, lw=0.2, edgecolor=INK)
    elif kind == "square":
        OV.add_patch(Rectangle((x - 0.9, y - 0.9), 1.8, 1.8, fc=c, ec="none", zorder=12))
    t(x + 2.4, y, s, size, ha="left")


# ------------------------------------------------------------------ icons (cx, cy, r, colour)
def ic_wave(cx, cy, r, c):
    xs = np.linspace(-1, 1, 60)
    OV.plot(cx + xs * r, cy + 0.6 * r * np.sin(xs * 9) * np.exp(-xs ** 2 * 2), color=c, lw=0.8, zorder=8)


def ic_clock(cx, cy, r, c):
    OV.add_patch(Circle((cx, cy), r, fc="white", ec=c, lw=0.7, zorder=8))
    OV.plot([cx, cx], [cy, cy + 0.6 * r], color=INK, lw=0.6, zorder=9)
    OV.plot([cx, cx + 0.45 * r], [cy, cy - 0.15 * r], color=INK, lw=0.6, zorder=9)


def ic_stopwatch(cx, cy, r, c):
    ic_clock(cx, cy - 0.1 * r, 0.85 * r, c)
    OV.add_patch(Rectangle((cx - 0.2 * r, cy + 0.78 * r), 0.4 * r, 0.3 * r, fc=c, ec="none", zorder=9))


def ic_head(cx, cy, r, c, n=12, ring=False):
    OV.add_patch(Circle((cx, cy), r, fc="white", ec=c, lw=0.6, zorder=8))
    OV.add_patch(Polygon([[cx - 0.2 * r, cy + 0.97 * r], [cx, cy + 1.25 * r], [cx + 0.2 * r, cy + 0.97 * r]],
                         fc="white", ec=c, lw=0.6, zorder=8))
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for px, py in [(0, 0), (0, 0.45), (0, -0.45)] + [(0.7 * np.cos(a), 0.7 * np.sin(a)) for a in ang]:
        OV.add_patch(Circle((cx + px * r, cy + py * r), 0.08 * r, fc=INK, zorder=9))
    if ring:
        OV.add_patch(Circle((cx, cy), 1.3 * r, fc="none", ec=GREY, lw=0.4, ls=(0, (2, 1.4)), zorder=8))


def ic_network(cx, cy, r, c):
    P = [(cx + r * np.cos(a), cy + 0.85 * r * np.sin(a)) for a in np.linspace(0.3, 2 * np.pi + 0.3, 6, endpoint=False)]
    for i, j in [(0, 2), (0, 3), (1, 4), (2, 5), (1, 3), (4, 5), (0, 5)]:
        OV.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color=c, lw=0.5, zorder=8)
    for p in P:
        OV.add_patch(Circle(p, 0.22 * r, fc=c, ec="white", lw=0.3, zorder=9))


def ic_bell(cx, cy, r, c):
    xs = np.linspace(-1, 1, 50); ys = np.exp(-xs ** 2 / 0.18)
    OV.plot(cx + xs * r, cy - 0.6 * r + ys * 1.3 * r, color=c, lw=0.7, zorder=9)
    m = xs > 0.55
    OV.fill_between(cx + xs[m] * r, cy - 0.6 * r, cy - 0.6 * r + ys[m] * 1.3 * r, color=TERRA, lw=0, zorder=9)
    OV.plot([cx - r, cx + r], [cy - 0.6 * r, cy - 0.6 * r], color=c, lw=0.5, zorder=9)


def ic_sliders(cx, cy, r, c):
    for i, f in enumerate((0.3, -0.4, 0.6)):
        yy = cy + (1 - i) * 0.7 * r
        OV.plot([cx - r, cx + r], [yy, yy], color=LG, lw=0.8, zorder=8)
        OV.add_patch(Circle((cx + f * r, yy), 0.28 * r, fc=c, ec="white", lw=0.3, zorder=9))


def ic_funnel(cx, cy, r, c):
    OV.add_patch(Polygon([[cx - r, cy + 0.8 * r], [cx + r, cy + 0.8 * r], [cx + 0.2 * r, cy - 0.1 * r],
                          [cx + 0.2 * r, cy - 0.85 * r], [cx - 0.2 * r, cy - 0.6 * r], [cx - 0.2 * r, cy - 0.1 * r]],
                         fc=c, ec="none", zorder=9))


def ic_speaker(cx, cy, r, c):
    OV.add_patch(Polygon([[cx - 0.9 * r, cy - 0.3 * r], [cx - 0.5 * r, cy - 0.3 * r], [cx, cy - 0.75 * r],
                          [cx, cy + 0.75 * r], [cx - 0.5 * r, cy + 0.3 * r], [cx - 0.9 * r, cy + 0.3 * r]],
                         fc=c, ec="none", zorder=9))
    for rr in (0.45, 0.8):
        OV.add_patch(Arc((cx + 0.1 * r, cy), 2 * rr * r, 2 * rr * r, theta1=-50, theta2=50, color=c, lw=0.6, zorder=9))


def face(cx, cy, r, fc="#F3D9C6", ec="#9C7B67", smile=0.6, z=8, brow=True):
    OV.add_patch(Ellipse((cx, cy), 1.7 * r, 2.05 * r, fc=fc, ec=ec, lw=0.5, zorder=z))
    for sx in (-1, 1):
        OV.add_patch(Circle((cx + sx * 0.35 * r, cy + 0.2 * r), 0.09 * r, fc=INK, zorder=z + 1))
        if brow:
            OV.plot([cx + sx * 0.2 * r, cx + sx * 0.5 * r], [cy + 0.42 * r, cy + 0.47 * r], color=INK, lw=0.5, zorder=z + 1)
    xs = np.linspace(-0.3, 0.3, 20) * r
    OV.plot(cx + xs, cy - 0.45 * r - smile * (0.09 * r - (xs / r) ** 2 * r), color="#8E3B2E", lw=0.6, zorder=z + 1)


def ic_face(cx, cy, r, c):
    face(cx, cy, 0.9 * r)


def ic_faceflow(cx, cy, r, c):
    face(cx - 0.2 * r, cy, 0.85 * r)
    for dy in (0.5, -0.5):
        arr(cx + 0.7 * r, cy + dy * r, cx + 1.25 * r, cy + dy * r * 1.3, c=c, lw=0.5, ms=2.5)


def ic_swap(cx, cy, r, c):
    OV.add_patch(Rectangle((cx - 1.1 * r, cy - 0.2 * r), 0.85 * r, 0.95 * r, fc="white", ec=c, lw=0.5, zorder=9))
    face(cx - 0.68 * r, cy + 0.28 * r, 0.3 * r, z=10, brow=False)
    face(cx + 0.7 * r, cy + 0.28 * r, 0.33 * r, z=10, smile=-0.8)
    arr(cx - 0.2 * r, cy - 0.6 * r, cx + 0.5 * r, cy - 0.6 * r, c=c, lw=0.5, ms=2.5)


def ic_transfer(cx, cy, r, c):
    rbox(cx - 1.15 * r, cy - 0.5 * r, 0.8 * r, 1.0 * r, fc="#DDE3E8", ec=c, lw=0.4, r=0.25 * r, z=9)
    face(cx + 0.65 * r, cy, 0.42 * r, z=9)
    arr(cx - 0.25 * r, cy + 0.75 * r, cx + 0.35 * r, cy + 0.75 * r, c=c, lw=0.5, ms=2.5)


def ic_eye(cx, cy, r, c):
    xs = np.linspace(-1, 1, 40)
    OV.fill_between(cx + xs * r, cy - 0.45 * r * (1 - xs ** 2), cy + 0.45 * r * (1 - xs ** 2), color="white",
                    ec=c, lw=0.6, zorder=8)
    OV.add_patch(Circle((cx, cy), 0.3 * r, fc=c, zorder=9))
    OV.add_patch(Circle((cx, cy), 0.12 * r, fc=INK, zorder=10))


def ic_ellipses(cx, cy, r, c):
    for dx, col in ((-0.45, TERRA), (0.45, TEAL)):
        OV.add_patch(Ellipse((cx + dx * r, cy + (0.2 if dx < 0 else -0.2) * r), 1.1 * r, 0.7 * r, angle=25, fc=col,
                             alpha=0.3, ec=col, lw=0.4, zorder=8))
    arr(cx - 0.45 * r, cy + 0.2 * r, cx + 0.45 * r, cy - 0.2 * r, c=INK, lw=0.5, ms=2.5, style="<|-|>")


def ic_bars(cx, cy, r, c):
    for i, (hh, col) in enumerate(((0.9, TEAL), (1.5, GREEN), (0.6, GOLD), (1.2, TERRA))):
        OV.add_patch(Rectangle((cx - r + i * 0.52 * r, cy - 0.8 * r), 0.4 * r, hh * r, fc=col, ec="none", zorder=9))


def ic_scale(cx, cy, r, c):
    OV.plot([cx, cx], [cy - 0.8 * r, cy + 0.6 * r], color=INK, lw=0.6, zorder=9)
    OV.plot([cx - r, cx + r], [cy + 0.5 * r, cy + 0.3 * r], color=INK, lw=0.6, zorder=9)
    OV.add_patch(Polygon([[cx - 1.2 * r, cy - 0.2 * r], [cx - 0.8 * r, cy - 0.2 * r], [cx - r, cy + 0.5 * r]],
                         fc=GREEN, ec="none", zorder=9))
    OV.add_patch(Polygon([[cx + 0.6 * r, cy - 0.4 * r], [cx + 1.0 * r, cy - 0.4 * r], [cx + 0.8 * r, cy + 0.3 * r]],
                         fc=TEAL, ec="none", zorder=9))


def ic_target(cx, cy, r, c):
    for rr, col in ((1.0, "white"), (0.66, TINT[c]), (0.33, c)):
        OV.add_patch(Circle((cx, cy), rr * r, fc=col, ec=c, lw=0.5, zorder=8))


def robot(cx, cy, s=1.0):
    OV.add_patch(Ellipse((cx, cy - 2.6 * s), 7.4 * s, 6.2 * s, fc="white", ec=GREY, lw=0.6, zorder=7))
    for sx in (-1, 1):
        OV.add_patch(Arc((cx + sx * 3.4 * s, cy - 2.6 * s), 2.4 * s, 5.2 * s, theta1=90 if sx < 0 else -90,
                         theta2=270 if sx < 0 else 90, color="#3D5AFE", lw=1.0, zorder=8))
    rbox(cx - 3.0 * s, cy - 0.6 * s, 6.0 * s, 4.6 * s, fc="#1C232B", ec=GREY, lw=0.6, r=1.2 * s, z=8)
    for sx in (-1, 1):
        OV.add_patch(Circle((cx + sx * 1.15 * s, cy + 1.7 * s), 0.55 * s, fc=CYAN, ec="none", zorder=9))
    OV.plot([cx, cx], [cy + 4.0 * s, cy + 5.0 * s], color=GREY, lw=0.6, zorder=7)
    OV.add_patch(Circle((cx, cy + 5.2 * s), 0.35 * s, fc=TERRA, ec="none", zorder=8))


def monitor(cx, cy, s=1.0):
    rbox(cx - 4.4 * s, cy - 2.2 * s, 8.8 * s, 6.0 * s, fc="#20272E", ec=GREY, lw=0.6, r=0.6, z=7)
    OV.add_patch(Rectangle((cx - 4.0 * s, cy - 1.8 * s), 8.0 * s, 5.2 * s, fc="#EAF4F6", ec="none", zorder=8))
    face(cx, cy + 0.8 * s, 1.9 * s, z=9)
    OV.add_patch(Polygon([[cx - 1.0 * s, cy - 2.2 * s], [cx + 1.0 * s, cy - 2.2 * s], [cx + 1.6 * s, cy - 3.4 * s],
                          [cx - 1.6 * s, cy - 3.4 * s]], fc=GREY, ec="none", zorder=7))


def ic_robot(cx, cy, r, c):
    robot(cx, cy - 0.05 * r, 0.21 * r)


def ic_monitor(cx, cy, r, c):
    monitor(cx, cy + 0.1 * r, 0.24 * r)


def check(cx, cy, r, ok=True):
    OV.add_patch(Circle((cx, cy), r, fc=GREEN if ok else TERRA, ec="white", lw=0.4, zorder=14))
    if ok:
        OV.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy, cy - 0.4 * r, cy + 0.4 * r], color="white", lw=0.7, zorder=15)
    else:
        OV.plot([cx - 0.4 * r, cx + 0.4 * r], [cy - 0.4 * r, cy + 0.4 * r], color="white", lw=0.7, zorder=15)
        OV.plot([cx - 0.4 * r, cx + 0.4 * r], [cy + 0.4 * r, cy - 0.4 * r], color="white", lw=0.7, zorder=15)


def tidy(a, grid=True):
    a.tick_params(length=2.2, pad=1.5)
    if grid:
        a.grid(axis="y", color="#E3E8EB", lw=0.5, zorder=0)
        a.set_axisbelow(True)


# ------------------------------------------------------------------ refined icons (iconkit) override the sketches above
def ic_head(cx, cy, r, c, n=12, ring=False):
    ik.head(cx, cy - 0.1 * r, 0.78 * r, z=8)
    if ring:
        OV.add_patch(Circle((cx, cy - 0.1 * r), 1.15 * r, fc="none", ec=GREY, lw=0.4, ls=(0, (2, 1.4)), zorder=7))


face = ik.face
robot = ik.robot
monitor = ik.monitor
check = ik.check
ic_wave = ik.wave
ic_clock = ik.clock
ic_stopwatch = ik.stopwatch
ic_network = ik.network
ic_bell = ik.bell
ic_sliders = ik.sliders
ic_funnel = ik.funnel
ic_speaker = ik.speaker
def ic_swap(cx, cy, r, c):
    ik.swap(cx - 0.2 * r, cy, 1.35 * r, c)


def ic_transfer(cx, cy, r, c):
    ik.transfer(cx - 0.2 * r, cy, 1.35 * r, c)


ic_eye = ik.eye
def ic_ellipses(cx, cy, r, c):
    ik.ellipses(cx - 0.2 * r, cy, 1.35 * r, c)


ic_bars = ik.bars
ic_scale = ik.scale
ic_target = ik.target
ic_faceflow = ik.faceflow


def ic_face(cx, cy, r, c):
    ik.face(cx, cy, 0.85 * r)


def ic_robot(cx, cy, r, c):
    ik.robot(cx, cy - 0.15 * r, 0.2 * r)


def ic_monitor(cx, cy, r, c):
    ik.monitor(cx, cy + 0.1 * r, 0.22 * r)
