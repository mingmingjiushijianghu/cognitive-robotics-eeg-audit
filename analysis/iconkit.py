"""Refined vector icons for the v3 figures (flat 2.0 style: two-tone shading, highlights, soft shadows).

All icons are drawn with matplotlib primitives on the axes stored in AX (millimetre coordinates);
no external images or third-party artwork. Shading uses clip paths: a darker side and a white
highlight are clipped to the base shape, and a soft ellipse below gives a contact shadow.
"""
import numpy as np
from matplotlib.patches import (FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, Arc, Wedge, PathPatch,
                                FancyArrowPatch)
from matplotlib.path import Path

AX = None
INK, GREY, LG = "#26313A", "#7D8A94", "#C9D1D7"
TEAL, TERRA, GOLD, PURP, GREEN, CYAN = "#1F7A8C", "#C0694E", "#B8954A", "#6E6AA6", "#3F7F5F", "#2EC4D6"
SKIN, SKIN_D, HAIR = "#F2D3BC", "#D9AE91", "#4A3B33"


def _add(p, z):
    p.set_zorder(z)
    AX.add_patch(p)
    return p


def shade(base, z, dark="#000000", dark_alpha=0.10, light_alpha=0.35, bbox=None, angle=0):
    """Darken the lower-right and highlight the upper-left of `base` (clipped)."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    d = _add(Polygon([[x0 + 0.55 * w, y1 + h], [x1 + w, y1 + h], [x1 + w, y0 - h], [x0 + 0.15 * w, y0 - h]],
                     fc=dark, ec="none", alpha=dark_alpha), z + 0.1)
    d.set_clip_path(base)
    hl = _add(Ellipse((x0 + 0.32 * w, y0 + 0.74 * h), 0.55 * w, 0.32 * h, angle=angle + 20, fc="white", ec="none",
                      alpha=light_alpha), z + 0.2)
    hl.set_clip_path(base)


def shadow(cx, cy, w, h, z, alpha=0.13):
    for k, a in ((1.0, alpha * 0.5), (0.7, alpha)):
        _add(Ellipse((cx, cy), w * k, h * k, fc="#1B2730", ec="none", alpha=a), z)


def rrect(x, y, w, h, r, fc, ec="none", lw=0.5, z=8):
    return _add(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw), z)


def glow(cx, cy, r, c, z):
    for k, a in ((2.2, 0.10), (1.6, 0.22), (1.0, 1.0)):
        _add(Circle((cx, cy), r * k, fc=c, ec="none", alpha=a), z)
    _add(Circle((cx - 0.3 * r, cy + 0.3 * r), 0.32 * r, fc="white", ec="none", alpha=0.9), z + 0.1)


# ------------------------------------------------------------------ characters
def face(cx, cy, r, fc=SKIN, ec="#9C7B67", smile=0.6, z=8, brow=True, hair=True):
    """Human face with hair, ears, eyes with highlights, brows, nose and mouth."""
    if hair:
        _add(Ellipse((cx, cy + 0.18 * r), 1.95 * r, 2.15 * r, fc=HAIR, ec="none"), z - 0.3)
    for sx in (-1, 1):
        _add(Ellipse((cx + sx * 0.86 * r, cy + 0.05 * r), 0.3 * r, 0.45 * r, fc=SKIN_D, ec="none"), z - 0.2)
    base = _add(Ellipse((cx, cy - 0.05 * r), 1.65 * r, 2.0 * r, fc=fc, ec=ec, lw=0.4), z)
    shade(base, z, dark="#8A5A3C", dark_alpha=0.12, light_alpha=0.30,
          bbox=(cx - 0.83 * r, cy - 1.05 * r, cx + 0.83 * r, cy + 0.95 * r))
    if hair:
        _add(PathPatch(Path([(cx - 0.86 * r, cy + 0.32 * r), (cx - 0.2 * r, cy + 1.25 * r), (cx + 0.86 * r, cy + 0.4 * r),
                             (cx + 0.3 * r, cy + 0.6 * r), (cx - 0.86 * r, cy + 0.32 * r)],
                            [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO]),
                       fc=HAIR, ec="none"), z + 0.3)
    for sx in (-1, 1):
        _add(Ellipse((cx + sx * 0.33 * r, cy + 0.08 * r), 0.24 * r, 0.17 * r, fc="white", ec="none"), z + 0.4)
        _add(Circle((cx + sx * 0.33 * r, cy + 0.08 * r), 0.075 * r, fc=INK, ec="none"), z + 0.5)
        _add(Circle((cx + sx * 0.33 * r + 0.03 * r, cy + 0.11 * r), 0.025 * r, fc="white", ec="none"), z + 0.6)
        if brow:
            AX.plot([cx + sx * 0.18 * r, cx + sx * 0.48 * r], [cy + 0.3 * r, cy + 0.33 * r], color=HAIR, lw=0.6,
                    solid_capstyle="round", zorder=z + 0.5)
        _add(Ellipse((cx + sx * 0.5 * r, cy - 0.35 * r), 0.28 * r, 0.14 * r, fc="#E8957B", ec="none", alpha=0.35), z + 0.4)
    AX.plot([cx, cx - 0.06 * r, cx + 0.03 * r], [cy - 0.02 * r, cy - 0.28 * r, cy - 0.3 * r], color="#B98A6E", lw=0.45,
            zorder=z + 0.5)
    xs = np.linspace(-0.26, 0.26, 20) * r
    AX.plot(cx + xs, cy - 0.55 * r - smile * (0.08 * r - (xs / r) ** 2 * r), color="#9C3F33", lw=0.6,
            solid_capstyle="round", zorder=z + 0.5)


def robot(cx, cy, s=1.0, z=7):
    """Desk companion robot with a screen face (Miko-like silhouette, generic design)."""
    shadow(cx, cy - 5.7 * s, 7.0 * s, 1.2 * s, z - 0.5)
    body = _add(Ellipse((cx, cy - 2.6 * s), 7.4 * s, 6.2 * s, fc="#F4F6F8", ec="#9AA6AF", lw=0.5), z)
    shade(body, z, dark="#5B6B78", dark_alpha=0.16, light_alpha=0.55,
          bbox=(cx - 3.7 * s, cy - 5.7 * s, cx + 3.7 * s, cy + 0.5 * s))
    for sx in (-1, 1):
        arc = Arc((cx + sx * 3.3 * s, cy - 2.6 * s), 2.2 * s, 5.0 * s, theta1=90 if sx < 0 else -90,
                  theta2=270 if sx < 0 else 90, color="#3D5AFE", lw=1.4, zorder=z + 0.3)
        AX.add_patch(arc)
        AX.add_patch(Arc((cx + sx * 3.3 * s, cy - 2.6 * s), 2.2 * s, 5.0 * s, theta1=90 if sx < 0 else -90,
                         theta2=270 if sx < 0 else 90, color="#A9B8FF", lw=0.45, zorder=z + 0.4))
    head = rrect(cx - 3.1 * s, cy - 0.7 * s, 6.2 * s, 4.8 * s, 1.3 * s, "#E9EDF0", ec="#9AA6AF", lw=0.5, z=z + 0.5)
    shade(head, z + 0.5, dark="#5B6B78", dark_alpha=0.14, light_alpha=0.5,
          bbox=(cx - 3.1 * s, cy - 0.7 * s, cx + 3.1 * s, cy + 4.1 * s))
    scr = rrect(cx - 2.55 * s, cy - 0.2 * s, 5.1 * s, 3.8 * s, 1.0 * s, "#141B22", ec="none", z=z + 0.8)
    refl = _add(Polygon([[cx - 2.6 * s, cy + 3.6 * s], [cx - 0.6 * s, cy + 3.6 * s], [cx - 2.6 * s, cy + 1.2 * s]],
                        fc="white", ec="none", alpha=0.10), z + 1.2)
    refl.set_clip_path(scr)
    for sx in (-1, 1):
        glow(cx + sx * 1.05 * s, cy + 1.6 * s, 0.48 * s, CYAN, z + 1.0)
    AX.plot([cx, cx], [cy + 4.1 * s, cy + 5.0 * s], color="#8B97A0", lw=0.7, zorder=z)
    glow(cx, cy + 5.25 * s, 0.32 * s, TERRA, z + 0.2)


def monitor(cx, cy, s=1.0, z=7, speech=True):
    """Desktop monitor showing a human face, with stand and speaker waves."""
    shadow(cx, cy - 3.7 * s, 6.5 * s, 0.9 * s, z - 0.5)
    _add(Polygon([[cx - 0.7 * s, cy - 2.2 * s], [cx + 0.7 * s, cy - 2.2 * s], [cx + 1.0 * s, cy - 3.3 * s],
                  [cx - 1.0 * s, cy - 3.3 * s]], fc="#8C979F", ec="none"), z)
    rrect(cx - 2.4 * s, cy - 3.7 * s, 4.8 * s, 0.55 * s, 0.25 * s, "#6F7A82", z=z)
    bez = rrect(cx - 4.5 * s, cy - 2.3 * s, 9.0 * s, 6.2 * s, 0.7 * s, "#262E36", ec="#11171C", lw=0.4, z=z + 0.2)
    shade(bez, z + 0.2, dark="#000000", dark_alpha=0.25, light_alpha=0.12,
          bbox=(cx - 4.5 * s, cy - 2.3 * s, cx + 4.5 * s, cy + 3.9 * s))
    scr = _add(Rectangle((cx - 4.05 * s, cy - 1.85 * s), 8.1 * s, 5.3 * s, fc="#DDEFF3", ec="none"), z + 0.4)
    _add(Rectangle((cx - 4.05 * s, cy - 1.85 * s), 8.1 * s, 1.9 * s, fc="#C7E3EA", ec="none"), z + 0.45).set_clip_path(scr)
    sh = _add(Ellipse((cx, cy - 2.0 * s), 4.6 * s, 2.6 * s, fc="#3B6E8F", ec="none"), z + 0.5)
    sh.set_clip_path(scr)
    face(cx, cy + 0.85 * s, 1.55 * s, z=z + 0.7)
    refl = _add(Polygon([[cx - 4.05 * s, cy + 3.45 * s], [cx - 1.0 * s, cy + 3.45 * s], [cx - 4.05 * s, cy + 0.2 * s]],
                        fc="white", ec="none", alpha=0.16), z + 1.6)
    refl.set_clip_path(scr)
    _add(Circle((cx, cy - 2.05 * s), 0.12 * s, fc="#5DD39E", ec="none"), z + 0.5)
    if speech:
        for rr in (0.7, 1.2, 1.7):
            AX.add_patch(Arc((cx + 4.7 * s, cy + 0.9 * s), rr * 2 * s, rr * 2.2 * s, theta1=-48, theta2=48, color=TEAL,
                             lw=0.6, zorder=z + 0.5, alpha=1.2 - rr / 2))


def humanoid(cx, cy, s=1.0, z=7):
    """Humanoid robot head with articulated parts and control sliders."""
    shadow(cx, cy - 3.9 * s, 5.0 * s, 0.8 * s, z - 0.5)
    rrect(cx - 1.1 * s, cy - 3.8 * s, 2.2 * s, 1.0 * s, 0.3 * s, "#AEB8C0", z=z)
    head = rrect(cx - 2.6 * s, cy - 3.0 * s, 5.2 * s, 6.4 * s, 2.3 * s, "#E3E8EC", ec="#8F9AA3", lw=0.45, z=z + 0.2)
    shade(head, z + 0.2, dark="#4E5C68", dark_alpha=0.18, light_alpha=0.5,
          bbox=(cx - 2.6 * s, cy - 3.0 * s, cx + 2.6 * s, cy + 3.4 * s))
    AX.plot([cx - 2.0 * s, cx + 2.0 * s], [cy + 2.2 * s, cy + 2.2 * s], color="#B5C0C8", lw=0.4, zorder=z + 0.5)
    for sx in (-1, 1):
        AX.plot([cx + sx * 0.5 * s, cx + sx * 1.5 * s], [cy + 1.55 * s, cy + 1.7 * s], color="#56636E", lw=0.7,
                solid_capstyle="round", zorder=z + 0.6)
        _add(Ellipse((cx + sx * 1.0 * s, cy + 0.75 * s), 1.35 * s, 0.75 * s, fc="white", ec="#56636E", lw=0.4), z + 0.6)
        _add(Circle((cx + sx * 1.0 * s, cy + 0.75 * s), 0.3 * s, fc="#5A86A8", ec="none"), z + 0.7)
        _add(Circle((cx + sx * 1.0 * s, cy + 0.75 * s), 0.13 * s, fc=INK, ec="none"), z + 0.8)
        _add(Circle((cx + sx * 1.0 * s + 0.08 * s, cy + 0.85 * s), 0.06 * s, fc="white", ec="none"), z + 0.9)
    AX.plot([cx, cx], [cy + 0.3 * s, cy - 0.6 * s], color="#B5C0C8", lw=0.5, zorder=z + 0.5)
    rrect(cx - 1.0 * s, cy - 1.9 * s, 2.0 * s, 0.6 * s, 0.3 * s, "#C9705E", ec="#9C4A3B", lw=0.3, z=z + 0.6)
    for k, f in enumerate((0.35, 0.75, 0.2)):
        yy = cy - 2.0 * s + k * 1.9 * s
        rrect(cx + 3.4 * s, yy - 0.18 * s, 3.2 * s, 0.36 * s, 0.18 * s, "#DCE2E6", z=z)
        rrect(cx + 3.4 * s, yy - 0.18 * s, 3.2 * s * f, 0.36 * s, 0.18 * s, GOLD, z=z + 0.1)
        _add(Circle((cx + 3.4 * s + 3.2 * s * f, yy), 0.42 * s, fc="white", ec=GOLD, lw=0.5), z + 0.2)


def headset(cx, cy, s, z=8):
    shadow(cx, cy - 1.6 * s, 5.4 * s, 0.6 * s, z - 0.5)
    AX.add_patch(Arc((cx, cy + 0.9 * s), 6.0 * s, 3.4 * s, theta1=0, theta2=180, color="#3B4650", lw=1.1, zorder=z - 0.2))
    body = rrect(cx - 2.9 * s, cy - 1.1 * s, 5.8 * s, 2.8 * s, 1.1 * s, "#3B4650", z=z)
    shade(body, z, dark="#000000", dark_alpha=0.25, light_alpha=0.18, bbox=(cx - 2.9 * s, cy - 1.1 * s, cx + 2.9 * s, cy + 1.7 * s))
    for sx in (-1, 1):
        lens = _add(Ellipse((cx + sx * 1.25 * s, cy + 0.3 * s), 1.9 * s, 1.35 * s, fc="#1C8FA6", ec="#0E5E6E", lw=0.3), z + 0.3)
        _add(Ellipse((cx + sx * 1.25 * s - 0.35 * s, cy + 0.55 * s), 0.7 * s, 0.35 * s, fc="white", alpha=0.55, ec="none"),
             z + 0.4).set_clip_path(lens)
    _add(Polygon([[cx - 0.35 * s, cy - 1.1 * s], [cx + 0.35 * s, cy - 1.1 * s], [cx, cy - 0.35 * s]], fc="#2A333B"), z + 0.3)


# ------------------------------------------------------------------ EEG / method icons
def head(cx, cy, r, n=16, c=GREY, fill="white", z=8, bad=None, cap=True, cable=False):
    """Top view of a head with EEG cap, electrodes (with rims) and nose/ears."""
    for sx in (-1, 1):
        _add(Ellipse((cx + sx * 1.0 * r, cy), 0.22 * r, 0.42 * r, fc=SKIN_D, ec="#9C7B67", lw=0.3), z - 0.2)
    _add(Polygon([[cx - 0.2 * r, cy + 0.96 * r], [cx, cy + 1.24 * r], [cx + 0.2 * r, cy + 0.96 * r]], fc=SKIN,
                 ec="#9C7B67", lw=0.3), z - 0.2)
    base = _add(Circle((cx, cy), r, fc=SKIN if cap else fill, ec="#9C7B67", lw=0.4), z)
    if cap:
        capc = _add(Circle((cx, cy), 0.93 * r, fc="#DDE7EC", ec="#8FA3AF", lw=0.35), z + 0.1)
        _add(Circle((cx, cy), 0.5 * r, fc="none", ec="#B9C8D0", lw=0.25), z + 0.15)
        AX.plot([cx - 0.93 * r, cx + 0.93 * r], [cy, cy], color="#B9C8D0", lw=0.25, zorder=z + 0.15)
        AX.plot([cx, cx], [cy - 0.93 * r, cy + 0.93 * r], color="#B9C8D0", lw=0.25, zorder=z + 0.15)
        shade(capc, z + 0.1, dark="#3E5866", dark_alpha=0.12, light_alpha=0.45, bbox=(cx - r, cy - r, cx + r, cy + r))
    ang = np.linspace(0, 2 * np.pi, 11, endpoint=False) + np.pi / 2
    pts = [(0, 0), (0, 0.42), (0, -0.42), (0.42, 0), (-0.42, 0)] + [(0.74 * np.cos(a), 0.74 * np.sin(a)) for a in ang]
    for i, (px, py) in enumerate(pts[:n]):
        isbad = bad is not None and i == bad
        _add(Circle((cx + px * r, cy + py * r), 0.11 * r, fc="#F7F9FA", ec="#5B6B78", lw=0.3), z + 0.3)
        _add(Circle((cx + px * r, cy + py * r), 0.06 * r, fc=TERRA if isbad else "#33424D", ec="none"), z + 0.4)
    AX.plot([cx + 0.15 * r, cx + 0.6 * r, cx + 1.25 * r], [cy - 0.98 * r, cy - 1.15 * r, cy - 1.05 * r],
            color="#5B6B78", lw=0.5, zorder=z - 0.3) if cable else None


def clock(cx, cy, r, c, z=9):
    shadow(cx + 0.1 * r, cy - 1.05 * r, 1.6 * r, 0.25 * r, z - 0.5)
    rim = _add(Circle((cx, cy), r, fc=c, ec="none"), z)
    shade(rim, z, dark="#000000", dark_alpha=0.22, light_alpha=0.3, bbox=(cx - r, cy - r, cx + r, cy + r))
    _add(Circle((cx, cy), 0.82 * r, fc="white", ec="none"), z + 0.3)
    for k, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        l0 = 0.6 if k % 3 == 0 else 0.68
        AX.plot([cx + l0 * r * np.cos(a), cx + 0.76 * r * np.cos(a)], [cy + l0 * r * np.sin(a), cy + 0.76 * r * np.sin(a)],
                color=INK, lw=0.55 if k % 3 == 0 else 0.3, zorder=z + 0.4)
    AX.plot([cx, cx + 0.05 * r], [cy, cy + 0.55 * r], color=INK, lw=0.8, solid_capstyle="round", zorder=z + 0.5)
    AX.plot([cx, cx + 0.38 * r], [cy, cy - 0.2 * r], color=INK, lw=0.8, solid_capstyle="round", zorder=z + 0.5)
    AX.plot([cx, cx - 0.2 * r], [cy, cy - 0.55 * r], color=TERRA, lw=0.4, zorder=z + 0.55)
    _add(Circle((cx, cy), 0.08 * r, fc=TERRA, ec="none"), z + 0.6)


def stopwatch(cx, cy, r, c, z=9):
    rrect(cx - 0.18 * r, cy + 0.8 * r, 0.36 * r, 0.32 * r, 0.06 * r, c, z=z)
    rrect(cx - 0.3 * r, cy + 1.05 * r, 0.6 * r, 0.16 * r, 0.06 * r, c, z=z)
    clock(cx, cy - 0.05 * r, 0.85 * r, c, z=z)


def speaker(cx, cy, r, c, z=9):
    body = _add(Polygon([[cx - 0.95 * r, cy - 0.32 * r], [cx - 0.5 * r, cy - 0.32 * r], [cx - 0.02 * r, cy - 0.8 * r],
                         [cx - 0.02 * r, cy + 0.8 * r], [cx - 0.5 * r, cy + 0.32 * r], [cx - 0.95 * r, cy + 0.32 * r]],
                        fc=c, ec="none"), z)
    shade(body, z, dark="#000000", dark_alpha=0.2, light_alpha=0.3, bbox=(cx - 0.95 * r, cy - 0.8 * r, cx, cy + 0.8 * r))
    for k, rr in enumerate((0.42, 0.72, 1.02)):
        AX.add_patch(Arc((cx + 0.05 * r, cy), 2 * rr * r, 2 * rr * r, theta1=-48, theta2=48, color=c, lw=0.8,
                         alpha=1.0 - 0.25 * k, zorder=z, capstyle="round"))


def funnel(cx, cy, r, c, z=9):
    body = _add(Polygon([[cx - r, cy + 0.75 * r], [cx + r, cy + 0.75 * r], [cx + 0.18 * r, cy - 0.12 * r],
                         [cx + 0.18 * r, cy - 0.8 * r], [cx - 0.18 * r, cy - 0.6 * r], [cx - 0.18 * r, cy - 0.12 * r]],
                        fc=c, ec="none"), z)
    shade(body, z, dark="#000000", dark_alpha=0.2, light_alpha=0.3, bbox=(cx - r, cy - 0.8 * r, cx + r, cy + 0.75 * r))
    _add(Ellipse((cx, cy + 0.75 * r), 2.0 * r, 0.35 * r, fc="white", ec=c, lw=0.4, alpha=0.9), z + 0.2)
    for dx, dy, rr in ((-0.45, 1.25, 0.12), (0.1, 1.45, 0.09), (0.5, 1.2, 0.1)):
        _add(Circle((cx + dx * r, cy + dy * r), rr * r, fc=c, ec="none", alpha=0.7), z)
    _add(Ellipse((cx, cy - 1.05 * r), 0.14 * r, 0.22 * r, fc="#5AA9C9", ec="none"), z)


def bell(cx, cy, r, c, z=9):
    xs = np.linspace(-1, 1, 60)
    ys = np.exp(-xs ** 2 / 0.16)
    base = cy - 0.6 * r
    AX.fill_between(cx + xs * r, base, base + ys * 1.3 * r, color=c, alpha=0.18, lw=0, zorder=z)
    m = xs > 0.5
    AX.fill_between(cx + xs[m] * r, base, base + ys[m] * 1.3 * r, color=TERRA, lw=0, zorder=z + 0.1)
    AX.plot(cx + xs * r, base + ys * 1.3 * r, color=c, lw=0.8, zorder=z + 0.2)
    AX.plot([cx - r, cx + r], [base, base], color=INK, lw=0.5, zorder=z + 0.2)
    for xx in (-0.5, 0, 0.5):
        AX.plot([cx + xx * r, cx + xx * r], [base, base - 0.1 * r], color=INK, lw=0.4, zorder=z + 0.2)
    AX.plot([cx + 0.5 * r, cx + 0.5 * r], [base, base + 0.95 * r], color=INK, lw=0.4, ls=(0, (1.5, 1)), zorder=z + 0.2)


def swap(cx, cy, r, c, z=9):
    rrect(cx - 1.15 * r, cy - 0.25 * r, 0.95 * r, 1.05 * r, 0.12 * r, "#262E36", z=z)
    _add(Rectangle((cx - 1.07 * r, cy - 0.17 * r), 0.79 * r, 0.89 * r, fc="#DDEFF3", ec="none"), z + 0.1)
    face(cx - 0.675 * r, cy + 0.27 * r, 0.3 * r, z=z + 0.2, brow=False)
    face(cx + 0.7 * r, cy + 0.27 * r, 0.36 * r, z=z + 0.2, smile=-0.9)
    AX.add_patch(FancyArrowPatch((cx - 0.15 * r, cy - 0.6 * r), (cx + 0.55 * r, cy - 0.6 * r), arrowstyle="-|>",
                                 mutation_scale=3, color=c, lw=0.6, zorder=z + 0.3))


def transfer(cx, cy, r, c, z=9):
    humanoid(cx - 0.8 * r, cy, 0.17 * r, z=z)
    face(cx + 0.75 * r, cy, 0.42 * r, z=z)
    AX.add_patch(FancyArrowPatch((cx - 0.2 * r, cy + 0.8 * r), (cx + 0.4 * r, cy + 0.8 * r), arrowstyle="-|>",
                                 mutation_scale=3, color=c, lw=0.6, zorder=z + 0.3))


def check(cx, cy, r, ok=True, z=12):
    shadow(cx, cy - 1.0 * r, 1.4 * r, 0.25 * r, z - 0.5, alpha=0.18)
    b = _add(Circle((cx, cy), r, fc=GREEN if ok else TERRA, ec="white", lw=0.5), z)
    shade(b, z, dark="#000000", dark_alpha=0.18, light_alpha=0.35, bbox=(cx - r, cy - r, cx + r, cy + r))
    if ok:
        AX.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy, cy - 0.38 * r, cy + 0.4 * r], color="white", lw=0.9,
                solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.5)
    else:
        for sgn in (1, -1):
            AX.plot([cx - 0.38 * r, cx + 0.38 * r], [cy - sgn * 0.38 * r, cy + sgn * 0.38 * r], color="white", lw=0.9,
                    solid_capstyle="round", zorder=z + 0.5)


def medal(cx, cy, r, z=8):
    for sx, col in ((-1, GREEN), (1, "#2F6A4D")):
        _add(Polygon([[cx + sx * 0.15 * r, cy - 0.2 * r], [cx + sx * 0.85 * r, cy - 1.6 * r], [cx + sx * 0.45 * r, cy - 1.45 * r],
                      [cx + sx * 0.25 * r, cy - 1.85 * r], [cx - sx * 0.25 * r, cy - 0.5 * r]], fc=col, ec="none"), z)
    shadow(cx + 0.1 * r, cy - 0.9 * r, 1.6 * r, 0.3 * r, z - 0.4, alpha=0.15)
    disc = _add(Circle((cx, cy + 0.2 * r), r, fc=GOLD, ec="#8E6F2E", lw=0.4), z + 0.2)
    shade(disc, z + 0.2, dark="#5A4210", dark_alpha=0.25, light_alpha=0.45, bbox=(cx - r, cy - 0.8 * r, cx + r, cy + 1.2 * r))
    _add(Circle((cx, cy + 0.2 * r), 0.7 * r, fc="none", ec="#FFF3D1", lw=0.5), z + 0.4)
    xs = np.linspace(-0.45, 0.45, 40)
    AX.plot(cx + xs * r, cy + 0.2 * r + 0.25 * r * np.sin(xs * 14) * np.exp(-xs ** 2 * 3), color="white", lw=0.7, zorder=z + 0.5)


def brain(cx, cy, r, c=PURP, z=9):
    """Lateral brain silhouette with gyri lines."""
    t = np.linspace(0, 2 * np.pi, 80)
    x = cx + r * (1.0 * np.cos(t) + 0.08 * np.cos(3 * t))
    y = cy + r * (0.72 * np.sin(t) + 0.06 * np.sin(5 * t)) + 0.1 * r * (np.cos(t) < -0.3) * (np.sin(t) < 0)
    b = _add(Polygon(np.c_[x, y], fc="#F2C7CF", ec="#B9707F", lw=0.4), z)
    shade(b, z, dark="#7A2E40", dark_alpha=0.15, light_alpha=0.35, bbox=(cx - r, cy - 0.8 * r, cx + r, cy + 0.8 * r))
    for k in range(5):
        xs = np.linspace(-0.75, 0.75, 30)
        yy = cy + (0.45 - 0.22 * k) * r + 0.08 * r * np.sin(xs * 9 + k)
        ln, = AX.plot(cx + xs * r, yy, color="#B9707F", lw=0.4, zorder=z + 0.3)
        ln.set_clip_path(b)
    _add(Ellipse((cx + 0.55 * r, cy - 0.75 * r), 0.5 * r, 0.3 * r, fc="#E7B3BE", ec="#B9707F", lw=0.4), z - 0.1)


def network(cx, cy, r, c, z=9):
    rng = np.random.default_rng(4)
    P = [(cx + r * 0.95 * np.cos(a), cy + r * 0.8 * np.sin(a)) for a in np.linspace(0.3, 2 * np.pi + 0.3, 7, endpoint=False)]
    P.append((cx, cy))
    for i, j in [(0, 2), (0, 3), (1, 4), (2, 5), (1, 3), (4, 6), (0, 6), (7, 1), (7, 4), (7, 5), (7, 0)]:
        AX.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color=c, lw=rng.uniform(0.35, 0.9), alpha=0.7, zorder=z)
    for k, p in enumerate(P):
        rr = 0.24 * r if k == 7 else 0.17 * r
        n = _add(Circle(p, rr, fc=c if k != 7 else TERRA, ec="white", lw=0.3), z + 0.2)
        _add(Circle((p[0] - 0.3 * rr, p[1] + 0.3 * rr), 0.35 * rr, fc="white", alpha=0.6, ec="none"), z + 0.3)


def eye(cx, cy, r, c, z=8):
    xs = np.linspace(-1, 1, 50)
    up, lo = cy + 0.5 * r * (1 - xs ** 2), cy - 0.45 * r * (1 - xs ** 2)
    white = _add(Polygon(np.r_[np.c_[cx + xs * r, up], np.c_[cx + xs[::-1] * r, lo[::-1]]], fc="white", ec=INK, lw=0.6), z)
    iris = _add(Circle((cx, cy), 0.36 * r, fc=c, ec="none"), z + 0.2); iris.set_clip_path(white)
    _add(Circle((cx, cy), 0.24 * r, fc="none", ec="white", lw=0.25, alpha=0.6), z + 0.25).set_clip_path(white)
    _add(Circle((cx, cy), 0.15 * r, fc=INK, ec="none"), z + 0.3).set_clip_path(white)
    _add(Circle((cx + 0.1 * r, cy + 0.12 * r), 0.06 * r, fc="white", ec="none"), z + 0.4)
    for xx in (-0.5, 0, 0.5):
        AX.plot([cx + xx * r, cx + xx * 1.2 * r], [cy + 0.5 * r * (1 - xx ** 2), cy + 0.5 * r * (1 - xx ** 2) + 0.22 * r],
                color=INK, lw=0.4, zorder=z + 0.1)


def wave(cx, cy, r, c, z=8):
    xs = np.linspace(-1, 1, 80)
    y = 0.15 * np.exp(-((xs + 0.45) / 0.07) ** 2) - 0.35 * np.exp(-((xs + 0.2) / 0.07) ** 2) + 0.75 * np.exp(-((xs + 0.0) / 0.12) ** 2) \
        + 0.3 * np.exp(-((xs - 0.45) / 0.22) ** 2)
    AX.fill_between(cx + xs * r, cy - 0.45 * r, cy - 0.45 * r + y * r, color=c, alpha=0.15, lw=0, zorder=z)
    AX.plot(cx + xs * r, cy - 0.45 * r + y * r, color=c, lw=0.9, solid_capstyle="round", zorder=z + 0.1)
    AX.plot([cx - r, cx + r], [cy - 0.45 * r, cy - 0.45 * r], color=INK, lw=0.35, zorder=z + 0.1)
    AX.plot([cx - 0.55 * r, cx - 0.55 * r], [cy - 0.6 * r, cy + 0.5 * r], color=INK, lw=0.35, zorder=z + 0.1)


def target(cx, cy, r, c, z=8):
    shadow(cx, cy - 1.05 * r, 1.5 * r, 0.25 * r, z - 0.5)
    for k, (rr, col) in enumerate(((1.0, c), (0.74, "white"), (0.5, c), (0.26, "white"))):
        _add(Circle((cx, cy), rr * r, fc=col, ec="none"), z + 0.1 * k)
    _add(Circle((cx, cy), 0.12 * r, fc=TERRA, ec="none"), z + 0.5)
    AX.add_patch(FancyArrowPatch((cx + 1.1 * r, cy + 1.0 * r), (cx + 0.1 * r, cy + 0.08 * r), arrowstyle="-|>",
                                 mutation_scale=3.5, color=INK, lw=0.6, zorder=z + 0.6))


def sliders(cx, cy, r, c, z=8):
    for i, f in enumerate((0.3, -0.4, 0.55)):
        yy = cy + (1 - i) * 0.72 * r
        rrect(cx - r, yy - 0.09 * r, 2 * r, 0.18 * r, 0.09 * r, "#DCE2E6", z=z)
        rrect(cx - r, yy - 0.09 * r, (f + 1) * r, 0.18 * r, 0.09 * r, c, z=z + 0.1)
        k = _add(Circle((cx + f * r, yy), 0.26 * r, fc="white", ec=c, lw=0.6), z + 0.2)


def ellipses(cx, cy, r, c, z=8):
    for dx, dy, col in ((-0.45, 0.22, TERRA), (0.45, -0.22, TEAL)):
        _add(Ellipse((cx + dx * r, cy + dy * r), 1.15 * r, 0.72 * r, angle=25, fc=col, alpha=0.28, ec=col, lw=0.5), z)
        _add(Ellipse((cx + dx * r, cy + dy * r), 0.6 * r, 0.38 * r, angle=25, fc="none", ec=col, lw=0.4, ls=(0, (1.5, 1))), z)
        _add(Circle((cx + dx * r, cy + dy * r), 0.1 * r, fc=col, ec="white", lw=0.3), z + 0.2)
    AX.add_patch(FancyArrowPatch((cx - 0.4 * r, cy + 0.2 * r), (cx + 0.4 * r, cy - 0.2 * r), arrowstyle="<|-|>",
                                 mutation_scale=2.8, color=INK, lw=0.5, zorder=z + 0.3))


def bars(cx, cy, r, c, z=8):
    AX.plot([cx - 1.1 * r, cx + 1.1 * r], [cy - 0.8 * r, cy - 0.8 * r], color=INK, lw=0.45, zorder=z + 0.3)
    for i, (hh, col) in enumerate(((0.9, TEAL), (1.6, GREEN), (0.6, GOLD), (1.25, TERRA))):
        b = rrect(cx - r + i * 0.52 * r, cy - 0.8 * r, 0.42 * r, hh * r, 0.06 * r, col, z=z)
        shade(b, z, dark="#000000", dark_alpha=0.15, light_alpha=0.3,
              bbox=(cx - r + i * 0.52 * r, cy - 0.8 * r, cx - r + i * 0.52 * r + 0.42 * r, cy - 0.8 * r + hh * r))


def scale(cx, cy, r, c, z=8):
    _add(Polygon([[cx - 0.35 * r, cy - 0.85 * r], [cx + 0.35 * r, cy - 0.85 * r], [cx, cy - 0.6 * r]], fc=INK), z)
    AX.plot([cx, cx], [cy - 0.65 * r, cy + 0.6 * r], color=INK, lw=0.7, zorder=z)
    AX.plot([cx - r, cx + r], [cy + 0.55 * r, cy + 0.35 * r], color=INK, lw=0.7, solid_capstyle="round", zorder=z)
    for sx, yy, col in ((-1, 0.55, GREEN), (1, 0.35, TEAL)):
        AX.plot([cx + sx * r, cx + sx * 0.75 * r], [cy + yy * r, cy - 0.15 * r + (yy - 0.45) * r], color=GREY, lw=0.35, zorder=z)
        AX.plot([cx + sx * r, cx + sx * 1.25 * r], [cy + yy * r, cy - 0.15 * r + (yy - 0.45) * r], color=GREY, lw=0.35, zorder=z)
        w = _add(Wedge((cx + sx * r, cy - 0.15 * r + (yy - 0.45) * r), 0.3 * r, 180, 360, fc=col, ec="none"), z + 0.1)
    _add(Circle((cx, cy + 0.6 * r), 0.1 * r, fc=GOLD, ec="none"), z + 0.2)


def faceflow(cx, cy, r, c, z=8):
    face(cx - 0.25 * r, cy, 0.8 * r, z=z)
    for dy in (0.45, -0.45):
        AX.add_patch(FancyArrowPatch((cx + 0.65 * r, cy + dy * r), (cx + 1.25 * r, cy + dy * r * 1.35), arrowstyle="-|>",
                                     mutation_scale=2.8, color=c, lw=0.6, zorder=z + 1))


def epochs(cx, cy, r, c, z=8, bad=2):
    """Stack of epoch cards with one rejected (autoreject)."""
    for k in range(4):
        x0 = cx - 1.05 * r + k * 0.55 * r
        card = rrect(x0, cy - 0.75 * r, 0.46 * r, 1.5 * r, 0.08 * r, "#F8D9CF" if k == bad else "#E3F1F3",
                     ec=TERRA if k == bad else "#8FB8C2", lw=0.35, z=z)
        xs = np.linspace(0, 1, 20)
        AX.plot(x0 + 0.04 * r + xs * 0.38 * r, cy + 0.1 * r * np.sin(xs * 12 + k) * (3 if k == bad else 1), color=TERRA if k == bad else TEAL,
                lw=0.45, zorder=z + 0.1)
    xb = cx - 1.05 * r + bad * 0.55 * r + 0.23 * r
    for sgn in (1, -1):
        AX.plot([xb - 0.28 * r, xb + 0.28 * r], [cy - sgn * 0.28 * r, cy + sgn * 0.28 * r], color=TERRA, lw=0.8,
                solid_capstyle="round", zorder=z + 0.3)


def components(cx, cy, r, c, z=8):
    """ICA component traces with the blink component removed."""
    cols = [PURP, TEAL, TERRA]
    for j in range(3):
        xs = np.linspace(-1, 1, 50)
        yy = cy + (0.6 - 0.6 * j) * r
        if j == 2:
            y = yy + 0.45 * r * np.exp(-((xs - 0.1) / 0.12) ** 2) - 0.1 * r
        else:
            y = yy + 0.16 * r * np.sin(xs * (8 + 4 * j) + j)
        AX.plot(cx + xs * r, y, color=cols[j], lw=0.6, zorder=z, alpha=0.55 if j == 2 else 1)
    AX.plot([cx - 1.05 * r, cx + 1.05 * r], [cy - 0.95 * r, cy - 0.25 * r], color=TERRA, lw=0.8, solid_capstyle="round", zorder=z + 0.2)
