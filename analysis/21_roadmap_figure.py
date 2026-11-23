"""Step 21: technical roadmap (Fig. 1) for the framework-structured manuscript.

Layered roadmap in the style of high-impact journal overview figures: numbered stages, three analysis
lanes with miniature schematic panels, integration column, and a validity gate. Every icon and mini
panel is drawn with matplotlib primitives (no external images, no generative imagery). Mini panels are
schematic; numeric results are shown in Figs. 2-8.
Output: submission/figures_v3/Fig1.png (600 dpi) and Fig1.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, Wedge, Arc,
                                FancyArrowPatch)
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "submission" / "figures_v3"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

INK, GREY, LG = "#26313A", "#7D8A94", "#C9D1D7"
TEAL, TERRA, GOLD, PURP, GREEN = "#1F7A8C", "#C0694E", "#B8954A", "#6E6AA6", "#3F7F5F"
CYAN = "#2EC4D6"
TINT = {TEAL: "#E3F1F3", TERRA: "#F8E9E3", GOLD: "#F6EFDF", PURP: "#ECEBF5", GREEN: "#E5F0E9", GREY: "#F1F4F6"}
W, H = 180.0, 140.0
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")


# ------------------------------------------------------------------ primitives
def rbox(x, y, w, h, fc="white", ec=GREY, lw=0.6, r=1.6, z=1, ls="-", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


def t(x, y, s, size=5.6, w="normal", c=INK, ha="center", va="center", z=10, style="normal", rot=0, ls=1.15):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, zorder=z, fontstyle=style,
            rotation=rot, linespacing=ls)


def arr(x0, y0, x1, y1, c=GREY, lw=0.8, ms=5, ls="-", rad=0.0, z=6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw,
                                 ls=ls, zorder=z, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def badge(x, y, n, c):
    ax.add_patch(Circle((x, y), 2.3, fc=c, ec="white", lw=0.6, zorder=12))
    t(x, y - 0.05, str(n), 6.2, "bold", "white", z=13)


def header(x, y, n, s, c):
    badge(x + 2.3, y, n, c)
    t(x + 5.6, y, s, 6.6, "bold", c, ha="left")


def inset(x, y, w, h):
    a = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=8)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    a.patch.set_alpha(0)
    return a


# ------------------------------------------------------------------ icons
def robot(cx, cy, s=1.0):
    ax.add_patch(Ellipse((cx, cy - 2.6 * s), 7.4 * s, 6.2 * s, fc="white", ec=GREY, lw=0.6, zorder=7))
    for sx in (-1, 1):
        ax.add_patch(Arc((cx + sx * 3.4 * s, cy - 2.6 * s), 2.4 * s, 5.2 * s, theta1=90 if sx < 0 else -90,
                         theta2=270 if sx < 0 else 90, color="#3D5AFE", lw=1.1, zorder=8))
    rbox(cx - 3.0 * s, cy - 0.6 * s, 6.0 * s, 4.6 * s, fc="#1C232B", ec=GREY, lw=0.6, r=1.2 * s, z=8)
    for sx in (-1, 1):
        ax.add_patch(Circle((cx + sx * 1.15 * s, cy + 1.7 * s), 0.55 * s, fc=CYAN, ec="none", zorder=9))
    ax.plot([cx, cx], [cy + 4.0 * s, cy + 5.0 * s], color=GREY, lw=0.6, zorder=7)
    ax.add_patch(Circle((cx, cy + 5.2 * s), 0.35 * s, fc=TERRA, ec="none", zorder=8))


def face(cx, cy, r, fc="#F3D9C6", ec="#9C7B67", smile=0.6, z=8, brow=True):
    ax.add_patch(Ellipse((cx, cy), 1.7 * r, 2.05 * r, fc=fc, ec=ec, lw=0.5, zorder=z))
    for sx in (-1, 1):
        ax.add_patch(Circle((cx + sx * 0.35 * r, cy + 0.2 * r), 0.09 * r, fc=INK, zorder=z + 1))
        if brow:
            ax.plot([cx + sx * 0.2 * r, cx + sx * 0.5 * r], [cy + 0.42 * r, cy + 0.47 * r], color=INK, lw=0.5, zorder=z + 1)
    xs = np.linspace(-0.3, 0.3, 20) * r
    ax.plot(cx + xs, cy - 0.45 * r - smile * (0.09 * r - (xs / r) ** 2 * r), color="#8E3B2E", lw=0.6, zorder=z + 1)


def monitor(cx, cy, s=1.0):
    rbox(cx - 4.4 * s, cy - 2.2 * s, 8.8 * s, 6.0 * s, fc="#20272E", ec=GREY, lw=0.6, r=0.6, z=7)
    ax.add_patch(Rectangle((cx - 4.0 * s, cy - 1.8 * s), 8.0 * s, 5.2 * s, fc="#EAF4F6", ec="none", zorder=8))
    face(cx, cy + 0.8 * s, 1.9 * s, z=9)
    ax.add_patch(Polygon([[cx - 1.0 * s, cy - 2.2 * s], [cx + 1.0 * s, cy - 2.2 * s], [cx + 1.6 * s, cy - 3.4 * s],
                          [cx - 1.6 * s, cy - 3.4 * s]], fc=GREY, ec="none", zorder=7))
    # speaker waves (AFFEC clips contain speech)
    for i, rr in enumerate((0.8, 1.4)):
        ax.add_patch(Arc((cx + 4.9 * s, cy + 0.8 * s), rr * 2 * s, rr * 2.2 * s, theta1=-50, theta2=50, color=TEAL,
                         lw=0.5, zorder=8))


def humanoid(cx, cy, s=1.0):
    rbox(cx - 2.6 * s, cy - 3.2 * s, 5.2 * s, 6.4 * s, fc="#DDE3E8", ec=GREY, lw=0.5, r=2.2 * s, z=7)
    for sx in (-1, 1):
        ax.add_patch(Ellipse((cx + sx * 1.0 * s, cy + 0.7 * s), 1.2 * s, 0.7 * s, fc="white", ec=INK, lw=0.4, zorder=8))
        ax.add_patch(Circle((cx + sx * 1.0 * s, cy + 0.7 * s), 0.25 * s, fc=INK, zorder=9))
    ax.plot([cx - 1.0 * s, cx + 1.0 * s], [cy - 1.6 * s, cy - 1.6 * s], color="#8E3B2E", lw=0.7, zorder=8)
    # control sliders
    for i in range(3):
        yy = cy - 2.2 * s + i * 2.2 * s
        ax.plot([cx + 3.6 * s, cx + 6.4 * s], [yy, yy], color=LG, lw=0.8, zorder=7)
        ax.add_patch(Circle((cx + (4.0 + 0.9 * i) * s, yy), 0.42 * s, fc=GOLD, ec="white", lw=0.3, zorder=8))


def head(cx, cy, r, n=16, c=GREY, fill="white", z=8, bad=None):
    ax.add_patch(Circle((cx, cy), r, fc=fill, ec=c, lw=0.5, zorder=z))
    ax.add_patch(Polygon([[cx - 0.18 * r, cy + 0.98 * r], [cx, cy + 1.22 * r], [cx + 0.18 * r, cy + 0.98 * r]],
                         fc=fill, ec=c, lw=0.5, zorder=z))
    ang = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    pts = [(0, 0), (0, 0.42), (0, -0.42), (0.42, 0), (-0.42, 0)] + [(0.75 * np.cos(a), 0.75 * np.sin(a)) for a in ang]
    for i, (px, py) in enumerate(pts[:n]):
        col = TERRA if bad is not None and i == bad else INK
        ax.add_patch(Circle((cx + px * r, cy + py * r), 0.07 * r if col == INK else 0.12 * r, fc=col, zorder=z + 1))


def clock(cx, cy, r, c):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=c, lw=0.7, zorder=9))
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.plot([cx + 0.78 * r * np.cos(a), cx + 0.92 * r * np.cos(a)], [cy + 0.78 * r * np.sin(a), cy + 0.92 * r * np.sin(a)],
                color=c, lw=0.4, zorder=10)
    ax.plot([cx, cx], [cy, cy + 0.6 * r], color=INK, lw=0.7, zorder=10)
    ax.plot([cx, cx + 0.45 * r], [cy, cy - 0.15 * r], color=INK, lw=0.7, zorder=10)


def speaker(cx, cy, r, c):
    ax.add_patch(Polygon([[cx - 0.9 * r, cy - 0.3 * r], [cx - 0.5 * r, cy - 0.3 * r], [cx, cy - 0.75 * r],
                          [cx, cy + 0.75 * r], [cx - 0.5 * r, cy + 0.3 * r], [cx - 0.9 * r, cy + 0.3 * r]],
                         fc=c, ec="none", zorder=9))
    for rr in (0.45, 0.8):
        ax.add_patch(Arc((cx + 0.1 * r, cy), 2 * rr * r, 2 * rr * r, theta1=-50, theta2=50, color=c, lw=0.6, zorder=9))


def funnel(cx, cy, r, c):
    ax.add_patch(Polygon([[cx - r, cy + 0.8 * r], [cx + r, cy + 0.8 * r], [cx + 0.2 * r, cy - 0.1 * r],
                          [cx + 0.2 * r, cy - 0.85 * r], [cx - 0.2 * r, cy - 0.6 * r], [cx - 0.2 * r, cy - 0.1 * r]],
                         fc=c, ec="none", zorder=9))


def bell(cx, cy, r, c):
    xs = np.linspace(-1, 1, 60)
    ys = np.exp(-xs ** 2 / 0.18)
    ax.plot(cx + xs * r, cy - 0.6 * r + ys * 1.3 * r, color=c, lw=0.7, zorder=9)
    m = xs > 0.55
    ax.fill_between(cx + xs[m] * r, cy - 0.6 * r, cy - 0.6 * r + ys[m] * 1.3 * r, color=TERRA, lw=0, zorder=9)
    ax.plot([cx - r, cx + r], [cy - 0.6 * r, cy - 0.6 * r], color=c, lw=0.5, zorder=9)


def swap(cx, cy, r, c):
    ax.add_patch(Rectangle((cx - r, cy - 0.15 * r), 0.75 * r, 0.9 * r, fc="white", ec=c, lw=0.5, zorder=9))
    face(cx - 0.62 * r, cy + 0.28 * r, 0.3 * r, z=10, brow=False)
    face(cx + 0.62 * r, cy + 0.28 * r, 0.3 * r, z=10, smile=-0.8)
    arr(cx - 0.15 * r, cy - 0.55 * r, cx + 0.45 * r, cy - 0.55 * r, c=c, lw=0.5, ms=3)


def transfer(cx, cy, r, c):
    rbox(cx - r, cy - 0.45 * r, 0.75 * r, 0.95 * r, fc="#DDE3E8", ec=c, lw=0.4, r=0.25 * r, z=9)
    face(cx + 0.6 * r, cy, 0.38 * r, z=9)
    arr(cx - 0.2 * r, cy + 0.65 * r, cx + 0.35 * r, cy + 0.65 * r, c=c, lw=0.5, ms=3)


def check(cx, cy, r, ok=True):
    ax.add_patch(Circle((cx, cy), r, fc=GREEN if ok else TERRA, ec="white", lw=0.4, zorder=12))
    if ok:
        ax.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy, cy - 0.4 * r, cy + 0.4 * r], color="white", lw=0.7, zorder=13)
    else:
        ax.plot([cx - 0.4 * r, cx + 0.4 * r], [cy - 0.4 * r, cy + 0.4 * r], color="white", lw=0.7, zorder=13)
        ax.plot([cx - 0.4 * r, cx + 0.4 * r], [cy + 0.4 * r, cy - 0.4 * r], color="white", lw=0.7, zorder=13)


def medal(cx, cy, r):
    ax.add_patch(Polygon([[cx - 0.5 * r, cy], [cx - 0.9 * r, cy - 1.5 * r], [cx - 0.4 * r, cy - 1.3 * r], [cx - 0.1 * r, cy - 1.7 * r],
                          [cx + 0.1 * r, cy - 1.7 * r], [cx + 0.4 * r, cy - 1.3 * r], [cx + 0.9 * r, cy - 1.5 * r], [cx + 0.5 * r, cy]],
                         fc=GREEN, ec="none", zorder=8))
    ax.add_patch(Circle((cx, cy + 0.2 * r), r, fc=GOLD, ec="white", lw=0.6, zorder=9))
    ax.add_patch(Circle((cx, cy + 0.2 * r), 0.62 * r, fc="none", ec="white", lw=0.5, zorder=10))
    xs = np.linspace(-0.45, 0.45, 30)
    ax.plot(cx + xs * r, cy + 0.2 * r + 0.25 * r * np.sin(xs * 14), color="white", lw=0.6, zorder=10)


def headset(cx, cy, s):
    rbox(cx - 2.8 * s, cy - 1.0 * s, 5.6 * s, 2.6 * s, fc="#3B4650", ec="none", r=1.0 * s, z=8)
    for sx in (-1, 1):
        ax.add_patch(Ellipse((cx + sx * 1.2 * s, cy + 0.3 * s), 1.7 * s, 1.2 * s, fc=CYAN, ec="none", alpha=0.85, zorder=9))
    ax.add_patch(Arc((cx, cy + 0.9 * s), 6.0 * s, 3.2 * s, theta1=0, theta2=180, color="#3B4650", lw=0.8, zorder=7))


# ------------------------------------------------------------------ refined icons (iconkit) replace the sketches above
import iconkit as ik
ik.AX = ax
robot, face, monitor, humanoid, head = ik.robot, ik.face, ik.monitor, ik.humanoid, ik.head
clock, speaker, funnel, bell, swap, transfer = ik.clock, ik.speaker, ik.funnel, ik.bell, ik.swap, ik.transfer
check, medal, headset = ik.check, ik.medal, ik.headset


# ------------------------------------------------------------------ mini panels (schematic)
TT = np.linspace(-0.2, 0.9, 200)


def erp_shape(tt, a170=-0.6, a300=1.2, alpp=0.6):
    return (a170 * np.exp(-((tt - 0.165) / 0.025) ** 2) + a300 * np.exp(-((tt - 0.33) / 0.06) ** 2) +
            alpp * np.exp(-((tt - 0.6) / 0.16) ** 2) + 0.25 * np.exp(-((tt - 0.1) / 0.02) ** 2))


def mini_erp(x, y, w, h):
    a = inset(x, y, w, h)
    for (lo, hi), c in zip([(0.13, 0.2), (0.25, 0.5), (0.4, 0.8)], ["#D7E3EA", "#F4E9CF", "#DCEBE2"]):
        a.axvspan(lo, hi, color=c, lw=0, alpha=0.9)
    a.plot(TT, erp_shape(TT), color=TEAL, lw=0.9)
    a.plot(TT, erp_shape(TT, -0.35, 0.8, 0.85) + 0.02 * np.sin(40 * TT), color=TERRA, lw=0.9)
    a.axhline(0, color=GREY, lw=0.3); a.axvline(0, color=GREY, lw=0.3)
    for xx, yy_, lab in ((0.165, -0.82, "N170"), (0.36, 1.6, "P300"), (0.66, 1.6, "LPP")):
        a.text(xx, yy_, lab, fontsize=3.7, ha="center", color=INK)
    a.set_ylim(-0.9, 1.85); a.set_xlim(-0.2, 0.9)


def mini_dots(x, y, w, h):
    a = inset(x, y, w, h)
    for i in range(3):
        for j, c in enumerate((TERRA, TEAL)):
            n = 10 if j == 0 else 30
            v = rng.normal([0.0, 0.1, -0.1][i] * (1 if j else 0), 0.08 if j == 0 else 0.3, n)
            a.scatter(i + (j - 0.5) * 0.4 + rng.uniform(-0.08, 0.08, n), v, s=0.8, color=c, alpha=0.5, lw=0)
            a.errorbar(i + (j - 0.5) * 0.4, v.mean(), yerr=v.std() / 3, fmt="o", ms=1.6, color=c, lw=0.5, capsize=0.8)
    a.axhline(0, color=GREY, lw=0.3)
    a.set_xlim(-0.6, 2.6); a.set_ylim(-1, 1)
    for i, l in enumerate(("N170", "P300", "LPP")):
        a.text(i, -1.25, l, fontsize=3.6, ha="center", color=INK)


def mini_latency(x, y, w, h):
    a = inset(x, y, w, h)
    yy = np.clip(erp_shape(TT), 0, None)
    cs = np.cumsum(yy); cs /= cs[-1]
    m = (TT > 0.25) & (TT < 0.5)
    a.fill_between(TT[m], 0, yy[m], color="#F4E9CF", lw=0)
    t50 = TT[m][np.argmin(np.abs(np.cumsum(yy[m]) / yy[m].sum() - 0.5))]
    a.fill_between(TT[m & (TT <= t50)], 0, yy[m & (TT <= t50)], color=GOLD, alpha=0.55, lw=0)
    a.plot(TT, erp_shape(TT), color=TEAL, lw=0.8)
    a.axvline(t50, color=TERRA, lw=0.6, ls="--")
    a.text(t50 + 0.02, 1.35, "50% area", fontsize=3.6, color=TERRA)
    a.axhline(0, color=GREY, lw=0.3)
    a.set_ylim(-0.9, 1.7); a.set_xlim(-0.2, 0.9)


def topo(a, seed, c1=(0.0, -0.5), c2=(0.3, 0.4)):
    g = np.linspace(-1.1, 1.1, 80)
    X, Y = np.meshgrid(g, g)
    Z = np.exp(-((X - c1[0]) ** 2 + (Y - c1[1]) ** 2) / 0.25) - 0.8 * np.exp(-((X - c2[0]) ** 2 + (Y - c2[1]) ** 2) / 0.2)
    im = a.imshow(Z, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    circ = Circle((0, 0), 1.0, transform=a.transData)
    im.set_clip_path(circ)
    a.add_patch(Circle((0, 0), 1.0, fc="none", ec=INK, lw=0.4))
    a.add_patch(Polygon([[-0.15, 0.98], [0, 1.18], [0.15, 0.98]], fc="none", ec=INK, lw=0.4))
    a.set_xlim(-1.25, 1.25); a.set_ylim(-1.15, 1.3); a.set_aspect("equal")


def mini_topos(x, y, w, h):
    a1 = inset(x, y, w / 2 - 0.3, h); topo(a1, 1)
    a2 = inset(x + w / 2 + 0.3, y, w / 2 - 0.3, h); topo(a2, 2, (0.0, 0.4), (-0.3, -0.5))
    t(x + w / 4, y - 1.0, "physical", 3.8, c=TERRA)
    t(x + 3 * w / 4, y - 1.0, "screen", 3.8, c=TEAL)


def mini_matrix(x, y, w, h):
    a = inset(x, y, w, h)
    M = rng.random((16, 16)) * 0.5
    M[:6, :6] += 0.45; M[11:, 11:] += 0.4; M[9:11, 9:11] += 0.5
    M = (M + M.T) / 2; np.fill_diagonal(M, 1)
    a.imshow(M, cmap="Purples", vmin=0, vmax=1.1)
    for s in a.spines.values():
        s.set_visible(True); s.set_linewidth(0.3); s.set_color(GREY)


def mini_graph(x, y, w, h):
    a = inset(x, y, w, h); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.45); a.set_aspect("equal")
    a.add_patch(Circle((0, 0), 1.15, fc="#F7F7FB", ec=GREY, lw=0.4))
    pos = {"F": [(-0.5, 0.75), (0, 0.85), (0.5, 0.75), (-0.25, 0.45), (0.25, 0.45)],
           "P": [(-0.4, -0.45), (0, -0.55), (0.4, -0.45), (0, -0.85)],
           "T": [(-0.95, 0.0), (0.95, 0.0)], "C": [(-0.45, 0.05), (0, 0.05), (0.45, 0.05)]}
    col = {"F": PURP, "P": TEAL, "T": GOLD, "C": LG}
    pts = [(p, k) for k, v in pos.items() for p in v]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (p1, k1), (p2, k2) = pts[i], pts[j]
            if {k1, k2} <= {"F", "P", "T"} and rng.random() < (0.55 if k1 != k2 else 0.3):
                a.plot([p1[0], p2[0]], [p1[1], p2[1]], color=PURP, lw=rng.uniform(0.2, 0.8), alpha=0.55)
    for p, k in pts:
        a.add_patch(Circle(p, 0.1, fc=col[k], ec="white", lw=0.3, zorder=5))


def mini_auc(x, y, w, h):
    a = inset(x, y, w, h)
    d = np.linspace(0.1, 0.5, 30)
    for k, (c, o) in enumerate(((PURP, 0.0), (TEAL, -0.08), (GOLD, -0.16))):
        v = 0.25 + 0.9 * d + o + 0.02 * np.sin(12 * d + k)
        a.plot(d, v, color=c, lw=0.7)
        if k == 0:
            a.fill_between(d, 0, v, color=c, alpha=0.12, lw=0)
    a.set_xlim(0.08, 0.52); a.set_ylim(0, 0.8)
    a.text(0.12, 0.7, "E", fontsize=3.8, color=PURP, style="italic"); a.text(0.2, 0.7, "C", fontsize=3.8, color=TEAL, style="italic")
    a.text(0.28, 0.7, "Q", fontsize=3.8, color=GOLD, style="italic")
    a.spines["bottom"].set_visible(True); a.spines["bottom"].set_linewidth(0.3)


def mini_nbs(x, y, w, h):
    a = inset(x, y, w, h); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.45); a.set_aspect("equal")
    a.add_patch(Circle((0, 0), 1.15, fc="#F7F7FB", ec=GREY, lw=0.4))
    P = [(-0.6, 0.6), (0.6, 0.6), (-0.9, -0.1), (0.9, -0.1), (-0.4, -0.7), (0.4, -0.7), (0, 0.1)]
    for i, j in [(0, 6), (6, 1), (6, 4), (4, 5), (2, 4)]:
        a.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color=TERRA, lw=0.9)
    for p in P:
        a.add_patch(Circle(p, 0.1, fc=INK, zorder=5))


def mini_mesh(x, y, w, h):
    a = inset(x, y, w, h); a.set_xlim(-1.2, 1.2); a.set_ylim(-1.35, 1.35); a.set_aspect("equal")
    a.add_patch(Ellipse((0, 0), 1.7, 2.2, fc="#F6E3D6", ec="#9C7B67", lw=0.4))
    pts = np.c_[rng.uniform(-0.7, 0.7, 70), rng.uniform(-0.95, 0.95, 70)]
    pts = pts[(pts[:, 0] / 0.82) ** 2 + (pts[:, 1] / 1.08) ** 2 < 1]
    a.scatter(pts[:, 0], pts[:, 1], s=0.35, color=GREEN, lw=0)
    for sx in (-1, 1):
        a.add_patch(Ellipse((sx * 0.35, 0.25), 0.35, 0.16, fc="white", ec=INK, lw=0.35))
    for (px, py), (dx, dy) in [((-0.35, -0.55), (-0.18, 0.12)), ((0.35, -0.55), (0.18, 0.12)), ((-0.35, 0.52), (0, 0.2)),
                               ((0.35, 0.52), (0, 0.2))]:
        a.annotate("", (px + dx, py + dy), (px, py), arrowprops=dict(arrowstyle="-|>", color=TERRA, lw=0.5, mutation_scale=3))
    a.plot(np.linspace(-0.35, 0.35, 10), -0.55 - 0.1 * (1 - (np.linspace(-1, 1, 10)) ** 2), color="#8E3B2E", lw=0.5)


def mini_kin(x, y, w, h):
    a = inset(x, y, w, h)
    tt = np.linspace(0, 3.5, 120)
    amp = 1 / (1 + np.exp(-(tt - 1.2) * 3)) * (1 - 0.15 * (tt - 1.5) ** 2 / 6)
    a.plot(tt, amp, color=GREEN, lw=0.8)
    a.plot(tt, np.gradient(amp, tt) * 0.8 + 0.05, color=GOLD, lw=0.7)
    a.plot(tt, 0.75 + 0.06 * np.sin(5 * tt), color=PURP, lw=0.6, ls="--")
    a.set_ylim(-0.05, 1.25)
    for yy, lab, c in ((1.15, "amplitude", GREEN), (0.92, "symmetry", PURP), (0.45, "speed", GOLD)):
        a.text(3.5, yy - 0.12, lab, fontsize=3.4, color=c, ha="right")
    a.spines["bottom"].set_visible(True); a.spines["bottom"].set_linewidth(0.3)


def mini_reg(x, y, w, h):
    a = inset(x, y, w, h)
    xx = rng.normal(0, 1, 40)
    a.scatter(xx, 0.45 * xx + rng.normal(0, 0.5, 40), s=0.9, color=GREEN, lw=0, alpha=0.8)
    a.scatter(xx, -2.3 + rng.normal(0, 0.5, 40), s=0.9, color=TEAL, lw=0, alpha=0.8)
    g = np.linspace(-2.2, 2.2, 5)
    a.plot(g, 0.45 * g, color=GREEN, lw=0.8); a.plot(g, -2.3 + 0 * g, color=TEAL, lw=0.8)
    a.text(-2.3, 1.55, "contagion", fontsize=3.6, color=GREEN); a.text(2.3, -3.6, "LPP", fontsize=3.6, color=TEAL, ha="right")
    a.set_xlim(-2.5, 2.5); a.set_ylim(-3.8, 2.2)


def mini_forest(x, y, w, h):
    a = inset(x, y, w, h)
    a.axvline(0, color=GREY, lw=0.4)
    for i, (m, lo, hi, c) in enumerate(((0.2, 0.05, 0.36, GREEN), (0.0, -0.09, 0.07, TEAL))):
        a.plot([lo, hi], [1 - i, 1 - i], color=c, lw=0.8)
        a.plot(m, 1 - i, "s", ms=2.0, color=c)
    a.set_xlim(-0.15, 0.42); a.set_ylim(-0.6, 1.6)
    a.text(0.21, 1.3, "subjective", fontsize=3.6, color=GREEN, ha="center")
    a.text(0.0, 0.3, "neural", fontsize=3.6, color=TEAL, ha="center")


# ================================================================== layout
# --- 0 header: motivation and the two research questions
rbox(2, 119.5, 176, 19.0, fc="#F3F6F8", ec=LG, lw=0.6, r=2.2)
ax.add_patch(Rectangle((2, 119.5), 2.2, 19.0, fc=INK, ec="none", zorder=2))
t(7, 135.4, "Does a robot need a physical body to convey emotion?", 7.0, "bold", INK, ha="left")
t(7, 131.6, "Emotion-expressing robots for care, education and companionship must choose between a physical face and a "
            "screen-mediated face; behavioral studies favor the physical robot, but its neural basis is unknown.",
  5.2, ha="left", style="italic")
for k, (x0, c, q, txt, route) in enumerate(((7.0, TEAL, "Q1", "First: does embodiment form change the EEG response, and at which stage?",
                                              "main evidence: Route 1"),
                                             (92.5, GREEN, "Q2", "Second: does subjective contagion follow the neural response?",
                                              "main evidence: Route 3"))):
    rbox(x0, 120.6, 83.5, 7.4, fc="white", ec=c, lw=0.5, r=1.4)
    t(x0 + 2.0, 124.3, q, 6.2, "bold", c, ha="left")
    t(x0 + 8.0, 125.6, txt, 5.0, ha="left")
    t(x0 + 8.0, 122.4, route, 4.5, c=c, ha="left", style="italic")
ik.brain(170.5, 133.4, 3.6)

# --- 1 public data
header(2, 116.0, 1, "Public datasets", TERRA)
cards = [(91.5, TERRA, "Miko · physical embodiment", ["co-present robot, screen eyes", "n = 10 · 16 ch · 128 Hz",
                                                                     "4 expressions, 4 s each"]),
         (66.5, TEAL, "AFFEC · screen-mediated", ["84 actor videos with speech", "n = 71 · 63 ch · 256 Hz",
                                                         "felt & perceived ratings"]),
         (41.5, GOLD, "X2C · expression space", ["100,000 humanoid faces", "30 control parameters",
                                                     "+ MediaPipe blendshapes"])]
for y0, c, title, lines in cards:
    rbox(2, y0, 40, 21.5, fc=TINT[c], ec=c, lw=0.6, r=1.8)
    ax.add_patch(Rectangle((2.0, y0 + 17.2), 40, 4.3, fc=c, ec="none", zorder=2, alpha=0.95))
    rbox(2, y0, 40, 21.5, fc="none", ec=c, lw=0.6, r=1.8, z=3)
    t(22, y0 + 19.35, title, 5.7, "bold", "white")
    for k, s in enumerate(lines):
        t(17.5, y0 + 13.4 - k * 4.0, s, 5.0, ha="left")
robot(9.5, 102.0, 1.05)
monitor(9.5, 77.5, 0.95)
humanoid(7.5, 51.5, 0.95)
# rating scale dots under AFFEC text
for k in range(9):
    ax.add_patch(Circle((18.5 + k * 2.2, 69.0), 0.55, fc=TEAL if k == 6 else "white", ec=TEAL, lw=0.3, zorder=9))
t(18.0, 69.0, "", 3)

# --- 2 harmonisation
header(45, 116.0, 2, "Preprocessing", INK)
rbox(45, 41.5, 22.5, 71.5, fc="#F7F9FA", ec=LG, lw=0.6, r=1.8)
steps = [("Resample 128 Hz", "0.5–60 Hz · notch"), ("PREP channels", "interpolate bad"),
         ("16-ch montage", "average reference"), ("Infomax ICA", "≤ 2 blink components"),
         ("autoreject", "±100 µV · ≥5 trials"), ("Baseline-SD units", "gain-free amplitudes")]
for k, (a_, b_) in enumerate(steps):
    yy = 106.5 - k * 11.6
    rbox(46.2, yy - 4.4, 20.1, 9.2, fc="white", ec=LG, lw=0.4, r=1.2, z=3)
    ix, iy = 49.6, yy + 0.2
    if k == 0:
        ik.wave(ix, iy, 2.4, TEAL)
    elif k == 1:
        head(ix, iy, 2.4, bad=7)
    elif k == 2:
        head(ix, iy, 2.4)
    elif k == 3:
        ik.components(ix, iy, 2.4, INK)
    elif k == 4:
        ik.epochs(ix, iy, 2.4, TEAL)
    else:
        ik.bell(ix, iy + 0.3, 2.4, INK)
    t(52.6, yy + 1.6, a_, 4.7, "bold", ha="left")
    t(52.6, yy - 1.9, b_, 4.1, c=GREY, ha="left")
    if k < 5:
        arr(56.2, yy - 4.45, 56.2, yy - 7.15, c=GREY, lw=0.5, ms=3)
# data -> preprocessing
for y0 in (102, 77, 52):
    arr(42.2, y0, 45.0, y0 if y0 != 52 else 52, c=GREY, lw=0.7, ms=4)

# --- 3 three analysis routes
header(70, 116.0, 3, "Analysis routes", INK)
lanes = [(90.0, TEAL, "Route 1 · main", "Neural response",
          [(mini_erp, "ERP components"), (mini_dots, "amplitude"), (mini_latency, "latency"), (mini_topos, "topography (TANOVA)")],
          "cell-mean LMM · permutation · FDR · TOST / Bayes factor"),
         (64.0, PURP, "Route 2 · support", "Brain network",
          [(mini_matrix, "dwPLI 16 × 16"), (mini_graph, "F–P–T modules"), (mini_auc, "E, C, Q (AUC 10–50%)"),
           (mini_nbs, "network-based statistic")], "θ / α / β · multitaper · 5-trial draws × 100"),
         (38.0, GREEN, "Route 3 · main", "Behavior–EEG",
          [(mini_mesh, "facial kinematics"), (mini_kin, "amplitude · speed · symmetry"), (mini_reg, "trial-level LMM"),
           (mini_forest, "subjective vs neural slope")], "crossed participant & clip effects · bootstrap · LOSO")]
for y0, c, r1, r2, panels, foot in lanes:
    rbox(70, y0, 80, 23.0, fc=TINT[c], ec=c, lw=0.6, r=1.8)
    ax.add_patch(Rectangle((70.0, y0), 7.0, 23.0, fc=c, ec="none", zorder=2))
    rbox(70, y0, 80, 23.0, fc="none", ec=c, lw=0.6, r=1.8, z=3)
    t(73.5, y0 + 11.5, f"{r1}\n{r2}", 5.6, "bold", "white", rot=90)
    for k, (fn, lab) in enumerate(panels):
        px = 78.6 + k * 17.9
        rbox(px, y0 + 5.6, 16.0, 16.0, fc="white", ec=LG, lw=0.4, r=1.0, z=4)
        fn(px + 1.0, y0 + 7.4, 14.0, 12.6)
        t(px + 8.0, y0 + 3.9, lab, 4.5, c=INK)
        if k < 3:
            arr(px + 16.1, y0 + 13.6, px + 17.8, y0 + 13.6, c=c, lw=0.6, ms=3.5)
    t(110, y0 + 1.5, foot, 4.3, c=GREY, style="italic")
# preprocessing -> lanes
for y0 in (101.5, 75.5, 49.5):
    arr(67.6, y0, 70.0, y0, c=GREY, lw=0.7, ms=4)

# --- 4 integration
header(153, 116.0, 4, "Integration", GOLD)
rbox(153, 77.0, 25, 36.0, fc=TINT[GOLD], ec=GOLD, lw=0.6, r=1.8)
t(165.5, 110.4, "Embodied emotional", 5.6, "bold")
t(165.5, 107.6, "distance (EED)", 5.6, "bold")
t(165.5, 104.9, "supporting measure", 4.6, c=GREY, style="italic")
for (cx, cy, c) in ((159.5, 97.0, TERRA), (171.0, 93.0, TEAL)):
    ax.add_patch(Ellipse((cx, cy), 8.5, 5.0, angle=25, fc=c, alpha=0.18, ec=c, lw=0.5, zorder=7))
    ax.add_patch(Ellipse((cx, cy), 4.6, 2.7, angle=25, fc="none", ec=c, lw=0.4, ls="--", zorder=7))
    pts = rng.multivariate_normal([cx, cy], [[2.2, 1.0], [1.0, 1.2]], 14)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.9, color=c, zorder=8, lw=0)
    ax.add_patch(Circle((cx, cy), 0.6, fc=c, ec="white", lw=0.3, zorder=9))
arr(160.2, 96.7, 170.3, 93.3, c=INK, lw=0.6, ms=3, style="<|-|>")
t(165.5, 97.4, "$D_M$", 5.6, c=INK)
t(165.5, 86.3, "$\\mathrm{EED}=\\sqrt{\\mathbf{d}^{\\top}\\hat{\\Sigma}^{-1}\\mathbf{d}}$", 6.2)
t(165.5, 82.4, "14 features · Ledoit–Wolf Σ", 4.4, c=GREY)
t(165.5, 79.6, "eye-opening corrected · 5,000 perm.", 4.4, c=GREY)

rbox(153, 46.0, 25, 28.5, fc=TINT[GREEN], ec=GREEN, lw=0.6, r=1.8)
medal(159.0, 66.0, 2.6)
t(169.6, 68.6, "Neural markers", 5.6, "bold")
t(169.6, 65.8, "of contagion", 5.6, "bold")
crit = ["FDR-significant slope", "LOSO sign stability", "physical gain ≠ 0"]
for k, s in enumerate(crit):
    ax.add_patch(Circle((156.4, 59.4 - k * 3.6), 0.9, fc="white", ec=GREEN, lw=0.4, zorder=8))
    t(156.4, 59.35 - k * 3.6, str(k + 1), 4.2, "bold", GREEN)
    t(158.2, 59.4 - k * 3.6, s, 4.6, ha="left")
t(165.5, 48.3, "subjective / neural slope ratio", 4.4, c=GREY, style="italic")

rbox(153, 25.0, 25, 18.5, fc="white", ec=INK, lw=0.6, r=1.8)
t(165.5, 40.6, "Morphology selection", 5.6, "bold")
robot(158.0, 32.6, 0.62)
rbox(162.4, 30.4, 6.2, 4.4, fc="#20272E", ec=GREY, lw=0.4, r=0.4, z=8)
ax.add_patch(Rectangle((162.8, 30.8), 5.4, 3.6, fc="#EAF4F6", zorder=9))
face(165.5, 32.6, 1.2, z=10)
headset(173.4, 32.6, 0.85)
for x_, s in ((158.0, "physical"), (165.5, "virtual"), (173.4, "mixed")):
    t(x_, 26.3, s, 4.4, c=GREY)

# lanes -> integration
arr(150.2, 101.5, 153.0, 101.5, c=TEAL, lw=0.7, ms=4)
arr(150.2, 75.5, 153.0, 84.0, c=PURP, lw=0.7, ms=4, rad=-0.2)
arr(150.2, 49.5, 153.0, 56.0, c=GREEN, lw=0.7, ms=4, rad=-0.2)
arr(165.5, 77.0, 165.5, 74.6, c=GOLD, lw=0.7, ms=4)
arr(165.5, 46.0, 165.5, 43.6, c=GREEN, lw=0.7, ms=4)

# --- 5 validity gate
header(2, 36.2, 5, "Validity gate for cross-form inference", TERRA)
rbox(2, 2.0, 148, 31.5, fc=TINT[TERRA], ec=TERRA, lw=0.6, r=2.0)
checks = [("C1", "Event timing", "ITC & GFP at markers\nvs shifted surrogates", clock),
          ("C2", "Event type", "face onset vs\nexpression change", swap),
          ("C3", "Auditory overlap", "speech onset vs\nERP windows", speaker),
          ("C4", "Montage validity", "16-ch vs 62-ch\nr ≥ .70", None),
          ("C5", "Kinematic transfer", "descriptor valid\nper face type", transfer),
          ("C6", "Preprocessing", "effects under two\ncleaning variants", funnel),
          ("C7", "Calibration", "false positives\nunder null data", bell)]
for k, (cid, name, test, icon) in enumerate(checks):
    cx = 12.6 + k * 20.6
    ax.add_patch(Circle((cx, 23.0), 6.0, fc="white", ec=TERRA, lw=0.7, zorder=8))
    if icon is None:
        head(cx, 23.0, 3.5)
        ax.add_patch(Circle((cx, 23.0), 4.6, fc="none", ec=GREY, lw=0.4, ls=(0, (2, 1.5)), zorder=9))
    else:
        icon(cx, 23.0, 4.0 if icon in (swap, transfer) else 3.2, TERRA if icon in (funnel, speaker) else INK)
    rbox(cx - 3.6, 28.0, 7.2, 3.2, fc=TERRA, ec="none", r=1.0, z=10)
    t(cx, 29.6, cid, 5.2, "bold", "white", z=11)
    t(cx, 14.6, name, 5.3, "bold")
    t(cx, 9.4, test, 4.4, c=GREY)
    if k < 6:
        arr(cx + 6.3, 23.0, cx + 14.3, 23.0, c=TERRA, lw=0.5, ms=3, ls=(0, (2, 1.5)))
t(76, 3.8, "Cross-form (physical vs screen) effects are interpreted only for measures that pass C1–C7;"
           " within-dataset effects require C3–C5 and C7.", 4.7, c=TERRA, style="italic")
# gate -> lanes and integration (dashed feedback)
for x_ in (80.0, 110.0, 140.0):
    arr(x_, 33.6, x_, 38.0, c=TERRA, lw=0.6, ms=3.5, ls=(0, (2, 1.5)))
arr(150.2, 11.5, 153.0, 11.5, c=TERRA, lw=0.7, ms=4)
rbox(153, 2.0, 25, 19.5, fc="white", ec=TERRA, lw=0.6, r=1.8)
t(165.5, 18.6, "Inference scope", 5.6, "bold", TERRA)
check(156.3, 14.4, 1.2, True); t(158.2, 14.4, "measure passes:", 4.5, ha="left")
t(158.2, 11.9, "cross-form inference", 4.5, ha="left", c=GREY)
check(156.3, 8.0, 1.2, False); t(158.2, 8.0, "measure fails:", 4.5, ha="left")
t(158.2, 5.5, "descriptive only", 4.5, ha="left", c=GREY)

fig.savefig(OUT / "Fig1.png", dpi=600, facecolor="white")
fig.savefig(OUT / "Fig1.pdf", facecolor="white")
print("saved", OUT / "Fig1.png")
