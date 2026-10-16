#!/usr/bin/env python3
"""
Imbalanced Data: Oversampling - Random Oversampling and SMOTE
Educational explainer video (about 2.5 minutes) rendered with matplotlib.

Install : pip install numpy matplotlib imageio imageio-ffmpeg
Render  : python make_oversampling_video.py                 -> oversampling_explainer.mp4
Options : --fps 20 --out my_video.mp4
Preview : python make_oversampling_video.py --preview frames/   (one PNG per scene, fast)

Everything is drawn in pixel coordinates (960 x 540) so each scene is easy to edit.
Change the list DUR to speed up / slow down any scene.
"""
import argparse, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Arc
from matplotlib.colors import to_rgba

W, H, PX = 960, 540, 0.72            # canvas size; 1 px = 0.72 pt at dpi=100
plt.rcParams.update({"lines.scale_dashes": False, "font.family": "DejaVu Sans"})

# ----------------------------------------------------------------- palette
BG, FG, MUT = "#f3f6fb", "#1b2438", "#667089"
BL, RD, GR = "#2f6fdb", "#e0453a", "#1f9d63"      # majority / minority / good
PN, LN, AC, OR = "#ffffff", "#d3dae7", "#1F3864", "#f0a020"

# -------------------------------------------------------------------- data
rng = np.random.default_rng(11)


def pts(n, cx, cy, sx, sy):
    p = np.c_[rng.normal(cx, sx, n), rng.normal(cy, sy, n)]
    return np.clip(p, [.04, .06], [.96, .94])


def smote(A, n, k=3):
    """SMOTE: x_new = x_i + lambda * (x_nn - x_i), nn chosen among the k nearest."""
    out = []
    for _ in range(n):
        i = int(rng.integers(len(A)))
        dist = np.linalg.norm(A - A[i], axis=1)
        dist[i] = np.inf
        nb = [int(m) for m in np.argsort(dist)[:k]]
        j, lam = int(rng.choice(nb)), float(rng.random())
        out.append(dict(i=i, j=j, l=lam, nb=nb, p=A[i] + lam * (A[j] - A[i])))
    return out


def P(S):
    return np.array([s["p"] for s in S])


MAJ = pts(100, .33, .5, .12, .17)           # 100 dots stand for 950 samples
MN = pts(10, .70, .5, .05, .10)             # 10 dots stand for 50 samples
ROS = rng.integers(0, 10, 90)               # indices of minority points to duplicate
S1 = smote(MN, 90)
S1P = P(S1)
MAJ2 = pts(80, .40, .5, .12, .16)           # overlapping-classes dataset
MN2 = np.vstack([pts(10, .64, .5, .06, .12), [[.27, .45], [.36, .60]]])   # last 2 = noise
S2 = smote(MN2, 18)
for s in S2:                                # "bad" = lands in a dense majority area
    s["bad"] = int((np.linalg.norm(MAJ2 - s["p"], axis=1) < .08).sum()) >= 4
TRMIN = MN[[i for i in range(10) if i % 5]]  # every 5th point goes to the test set
TRSYN = P(smote(TRMIN, 40))

# ---------------------------------------------------------- drawing helpers
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])


def cl(v, a=0., b=1.): return min(b, max(a, v))
def sg(p, a, b): return cl((p - a) / (b - a))
def ez(t): return t * t * (3 - 2 * t)
def f_(p): return lambda a, b: ez(sg(p, a, b))


def mp(R, A):
    A = np.asarray(A, float).reshape(-1, 2)
    return np.c_[R[0] + A[:, 0] * R[2], R[1] + A[:, 1] * R[3]]


def T(s, x, y, sz=18, c=FG, al="left", a=1., w="bold", z=6):
    if a > .01:
        ax.text(x, y, str(s), fontsize=sz * PX, color=c, ha=al, va="center",
                alpha=a, fontweight=w, zorder=z)


def dots(Q, c, r=5., a=1., z=3):
    Q = np.asarray(Q, float)
    if Q.size == 0: return
    al = np.broadcast_to(a, (len(Q),))
    ax.scatter(Q[:, 0], Q[:, 1], s=(2 * r * PX) ** 2, c=[to_rgba(c, x) for x in al],
               linewidths=0, zorder=z)


def dias(Q, c, r=6.5, a=1.):
    Q = np.asarray(Q, float)
    if Q.size == 0: return
    al = np.broadcast_to(a, (len(Q),))
    ax.scatter(Q[:, 0], Q[:, 1], s=(1.414 * r * PX) ** 2, marker="D",
               c=[to_rgba(c, x) for x in al], edgecolors=PN, linewidths=1.0, zorder=3.5)


def ring(x, y, r, c, a=1., w=2.):
    if a > .01:
        ax.add_patch(Circle((x, y), max(r, .1), fill=False, ec=to_rgba(c, a), lw=w * PX, zorder=5))


def line(x1, y1, x2, y2, c, w=2., dash=None, a=1.):
    ls = (0, tuple(d * PX for d in dash)) if dash else "-"
    ax.plot([x1, x2], [y1, y2], color=c, lw=w * PX, ls=ls, alpha=a, zorder=4)


def rect(x, y, w, h, c, a=1., z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=to_rgba(c, a), ec="none", zorder=z))


def box(R, title="", a=1.):
    x, y, w, h = R
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                fc=to_rgba(BG, a), ec=to_rgba(LN, a), lw=1.5 * PX, zorder=1))
    if title: T(title, x + 12, y - 14, 14, MUT, a=a, w="normal")


def pb(R, fn, c, dash, a):
    ys = np.linspace(0, 1, 21)
    xs = np.array([fn(y) for y in ys])
    ax.plot(R[0] + xs * R[2], R[1] + ys * R[3], color=c, lw=2.5 * PX, alpha=a, zorder=4,
            ls=(0, tuple(d * PX for d in dash)) if dash else "-")


def mj(R, a=1.): dots(mp(R, MAJ), BL, 4.5, a)
def mi(R, a=1., A=None): dots(mp(R, MN if A is None else A), RD, 6, a)


def cn(a, b, x=670, y=140):
    T("Majority", x, y, 15, MUT, w="normal"); T(a, x, y + 34, 40, BL, w="heavy")
    T("Minority", x, y + 95, 15, MUT, w="normal"); T(b, x, y + 129, 40, RD, w="heavy")


R1 = (60, 100, 560, 400)

# ------------------------------------------------------------------ scenes
def s0(p):                                           # 1  The Imbalanced Data Problem
    f = f_(p); box(R1, "Feature space (2D scatter plot)")
    bx = R1[0] + .78 * R1[2]
    if p > .55:
        e = f(.55, .65)
        rect(R1[0], R1[1], bx - R1[0], R1[3], BL, .07 * e)
        rect(bx, R1[1], R1[0] + R1[2] - bx, R1[3], RD, .07 * e)
        T("predicted: Majority", R1[0] + 10, R1[1] + R1[3] - 16, 13, BL, a=e, w="normal")
        T("predicted: Minority", R1[0] + R1[2] - 10, R1[1] + R1[3] - 16, 13, RD, "right", e, "normal")
    dots(mp(R1, MAJ), BL, 4.5, [f(.14 + i * .0015, .2 + i * .0015) for i in range(100)])
    dots(mp(R1, MN), RD, 6, [f(.32 + i * .012, .38 + i * .012) for i in range(10)])
    if p > .55:
        line(bx, R1[1], bx, R1[1] + R1[3] * f(.55, .65), FG, 2.5, [7, 5])
        T("decision boundary", bx - 6, R1[1] + 16, 13, FG, "right", f(.6, .7), "normal")
    for q in mp(R1, MN):
        if (q[0] - R1[0]) / R1[2] < .78 and p > .72:
            ring(q[0], q[1], 10 + 2 * np.sin(p * 60), RD, f(.72, .8))
    cn(round(950 * f(.14, .32)), round(50 * f(.32, .45)))
    T("950 Majority vs 50 Minority", 670, 300, 16, FG, a=f(.45, .52))
    if p > .72:
        T("✗ Many minority samples", 670, 345, 16, RD, a=f(.72, .8)); T("are missed", 670, 368, 16, RD, a=f(.72, .8))
        T("Model is biased toward", 670, 415, 18, AC, a=f(.8, .88)); T("the majority class", 670, 440, 18, AC, a=f(.8, .88))
    a = 1 - sg(p, .09, .15)                          # title card
    if a > 0:
        rect(0, 0, W, H, PN, a, z=11)
        T("Imbalanced Data: Oversampling", 480, 225, 42, AC, "center", a, "heavy", 12)
        T("Random Oversampling and SMOTE", 480, 285, 28, MUT, "center", a, "normal", 12)
        T("Applied Machine Learning  ·  Mid-Sem Project", 480, 345, 16, MUT, "center", a, "normal", 12)


def s1(p):                                           # 2  Why accuracy can be misleading
    f = f_(p); gx, gy, cw = 130, 120, 14
    idx = np.arange(1000)
    al = [ez(sg(p, i / 4e3, i / 4e3 + .08)) for i in idx]
    ax.scatter(gx + idx % 50 * cw + 5.5, gy + idx // 50 * cw + 5.5, s=(11 * PX) ** 2, marker="s",
               c=[to_rgba(RD if i in (333, 777) else BL, a) for i, a in zip(idx, al)],
               linewidths=0, zorder=3)
    T("1,000 transactions", gx, gy - 18, 16, MUT, w="normal")
    T("■ genuine (998)", gx + 250, gy - 18, 14, BL); T("■ fraud (2)", gx + 420, gy - 18, 14, RD)
    s = sg(p, .35, .62); sx = gx + 700 * s
    if s > 0:
        rect(gx - 4, gy - 4, sx - gx + 4, 288, GR, .22)
        line(sx, gy - 6, sx, gy + 286, GR, 3)
        T('Model predicts "Genuine" for every transaction', 480, 418, 17, GR, "center", f(.4, .5))
    if p > .62:
        e = f(.62, .72)
        T(f"Accuracy: {99.8 * e:.1f}%", 130, 458, 28, GR, a=e, w="heavy")
        T("Fraud detected: 0 / 2", 520, 458, 28, RD, a=e, w="heavy")
        ring(gx + 33 * cw + 5.5, gy + 6 * cw + 5.5, 12, RD, f(.66, .74))
        ring(gx + 27 * cw + 5.5, gy + 15 * cw + 5.5, 12, RD, f(.66, .74))
    if p > .8: T("High Accuracy ≠ Good Minority Detection", 480, 505, 28, AC, "center", f(.8, .9), "heavy")


def s2(p):                                           # 3  Oversampling
    f = f_(p); box(R1, "Training data"); mj(R1); mi(R1)
    q = f(.2, .75); n = int(90 * q)
    dots(mp(R1, S1P[:n]), RD, 5.5, .9)
    cn(950, 50 + round(900 * q))
    if p > .6:
        e = f(.6, .7)
        T("Add minority samples", 670, 330, 21, AC, a=e); T("to the TRAINING DATA", 670, 360, 21, AC, a=e)
        T("→ classes become balanced", 670, 405, 16, MUT, a=e, w="normal")


def s3(p):                                           # 4  Random Oversampling
    f = f_(p); box(R1, "Random Oversampling"); mj(R1, .8)
    cc = np.zeros(10, int); landed, flying, fly = [], [], None
    for c, i in enumerate(ROS):
        u = (p - (.14 + c / 90 * .5)) / .03
        x, y = mp(R1, MN[i])[0]
        if u >= 1: cc[i] += 1; landed.append((x, y))
        elif u > 0:
            ox = 30 * (1 - ez(u)); flying.append((x + ox, y - ox)); fly = (x + ox + 12, y - ox - 12)
    dots(landed, RD, 6, .5); dots(flying, RD, 6, .8); mi(R1)
    for i, q in enumerate(mp(R1, MN)):
        if cc[i]: ring(q[0], q[1], 7 + min(cc[i], 20) * .6, RD, .5, 1.5)
        if p < .14: ring(q[0], q[1], 10 + 5 * np.sin(p * 80), RD, 1)
    if fly: T("copy", fly[0], fly[1], 12, MUT, w="normal")
    cn(950, 50 + int(cc.sum()) * 10)
    if p > .62:
        e = f(.62, .7)
        for k, s in enumerate(["Random Oversampling =", "Duplicate existing", "minority samples"]):
            T(s, 670, 305 + 25 * k, 18, FG, a=e)
    if p > .8:
        e = f(.8, .9)
        T("✓ Simple and fast", 670, 400, 17, GR, a=e, w="normal")
        T("✗ Adds no new information", 670, 432, 17, RD, a=e, w="normal")
        T("and may cause overfitting", 670, 455, 17, RD, a=e, w="normal")


def st(j): return (.04 + j * .115, .115) if j < 5 else (.615 + (j - 5) * .0105, .0105)


def s4(p):                                           # 5  SMOTE
    box(R1, "SMOTE"); mj(R1, .55)
    cur, done, fast = None, [], []
    for j, s in enumerate(S1[:30]):
        a, d = st(j); u = (p - a) / d
        if u >= 1: done.append(s["p"])
        elif u > 0:
            if j < 5: cur = (s, u)
            else: fast.append((s["p"], u))
    dias(mp(R1, done), RD)
    for q, u in fast: dias(mp(R1, q), RD, 6.5, u)
    mi(R1)
    if cur:
        s, u = cur
        x1, y1 = mp(R1, MN[s["i"]])[0]; x2, y2 = mp(R1, MN[s["j"]])[0]
        ring(x1, y1, 11 + 2 * np.sin(u * 20), RD, sg(u, 0, .1))
        for m in s["nb"]:
            x, y = mp(R1, MN[m])[0]; ring(x, y, 10, GR, sg(u, .1, .25) * (1 if m == s["j"] else .55))
        e = sg(u, .3, .5)
        if e > 0: line(x1, y1, x1 + (x2 - x1) * e, y1 + (y2 - y1) * e, FG, 2, [6, 5], 1 - sg(u, .85, 1))
        m = ez(sg(u, .5, .8))
        if m > 0:
            x, y = mp(R1, s["p"])[0]
            dias([(x, y)], RD, 7 * m); ring(x, y, 6 + 18 * (1 - m), GR, 1 - m)
            T(r"$x_{new}$", x + 12, y - 14, 13, GR, a=m)
        T("1. pick x_i, find its k nearest neighbours" if u < .3 else "2. choose one neighbour x_nn" if u < .5
          else "3. new point on the line", 670, 345, 15, FG, w="normal")
        T(f"λ = {s['l']:.2f}", 670, 305, 22, GR, a=sg(u, .4, .55))
    T("● original minority", 670, 140, 15, RD); T("◆ synthetic (new)", 670, 168, 15, RD)
    T(r"$x_{new}=x_i+\lambda\,(x_{nn}-x_i)$", 670, 228, 17, AC)
    T("0 ≤ λ ≤ 1,   k = 3 neighbours", 670, 256, 14, MUT, w="normal")
    T(f"Synthetic points: {len(done)}", 670, 400, 16, MUT, w="normal")
    if p > .88: T("New points — not duplicates ✓", 670, 440, 16, GR, a=ez(sg(p, .88, .95)))


def s5(p):                                           # 6  ROS vs SMOTE (main focus)
    Rs = [(40, 110, 430, 330), (490, 110, 430, 330)]
    q = ez(sg(p, .2, .7)); n = int(90 * q); bm = ez(sg(p, .7, .9)); al = ez(sg(p, .05, .15))
    for k, R in enumerate(Rs):
        box(R, "SMOTE" if k else "Random Oversampling"); mj(R, .55)
        if k: dias(mp(R, S1P[:n]), RD, 5.5, .85)
        else:
            cc = np.bincount(ROS[:n], minlength=10)
            for i, m in enumerate(mp(R, MN)):
                if cc[i]: ring(m[0], m[1], 7 + min(cc[i], 20) * .6, RD, .5, 1.5)
        mi(R)
        b0, b1, cv = .78, (.6 if k else .66), (-.07 if k else 0)
        if bm > 0: pb(R, lambda y: b0, MUT, [4, 4], .8)
        pb(R, lambda y: b0 + (b1 - b0) * bm + cv * bm * np.cos((y - .5) * np.pi), FG, None, al)
        T("boundary: before (grey) → after", R[0] + R[2] - 8, R[1] + R[3] - 14, 12, MUT, "right", al, "normal")
        T(f"Distinct minority locations: {10 + n if k else 10}", R[0] + 8, R[1] + R[3] + 18, 14, MUT, w="normal")
    T("ROS → Copies existing samples", 255, 488, 19, BL, "center")
    T("SMOTE → Creates synthetic samples", 705, 488, 19, RD, "center")
    T("Illustrative decision boundaries: balancing the training data moves the boundary toward the minority region.",
      480, 520, 12, MUT, "center", w="normal")


def s6(p):                                           # 7  Important limitation
    f = f_(p); box(R1, "Overlapping classes + noisy points")
    dots(mp(R1, MAJ2), BL, 4.5, .8); dots(mp(R1, MN2), RD, 6)
    for i in (10, 11):
        x, y = mp(R1, MN2[i])[0]; T("noise", x + 10, y - 13, 12, MUT, a=sg(p, .1, .2), w="normal")
    sl = S2[:int(len(S2) * sg(p, .15, .7))]
    for s in sl:
        a, b = mp(R1, MN2[s["i"]])[0], mp(R1, MN2[s["j"]])[0]
        line(a[0], a[1], b[0], b[1], MUT, 1, [3, 4], .35)
    for s in sl:
        x, y = mp(R1, s["p"])[0]; bad = s["bad"] and p > .7
        dias([(x, y)], OR if bad else RD)
        if bad: ring(x, y, 12, OR, sg(p, .7, .78))
    T("Classes overlap and", 670, 150, 17, FG, w="normal"); T("minority data has noise", 670, 175, 17, FG, w="normal")
    if p > .7:
        e = sg(p, .7, .78)
        T("◆ orange = synthetic points", 670, 240, 16, OR, a=e, w="normal")
        T("landing in majority territory", 670, 265, 16, OR, a=e, w="normal")
    if p > .82:
        e = f(.82, .92)
        T("SMOTE is useful,", 670, 350, 20, AC, a=e, w="heavy"); T("but not always perfect.", 670, 382, 20, AC, a=e, w="heavy")


FB, TB, XB = (50, 130, 860, 300), (50, 130, 560, 300), (650, 130, 260, 300)


def s7(p):                                           # 8  Very important rule
    f = f_(p); sp = f(.12, .38)
    box(FB, "Original dataset", 1 - sp); box(TB, "Training data (80%)", sp); box(XB, "Test data (20%)", sp)

    def pos(A, test):
        a = mp(FB, A); b = np.where(test[:, None], mp(XB, A), mp(TB, A))
        return a + (b - a) * sp
    tm = np.arange(100) % 5 == 0; tn = np.arange(10) % 5 == 0
    dots(pos(MAJ, tm), BL, 4); dots(pos(MN, tn), RD, 5.5)
    n = int(len(TRSYN) * sg(p, .42, .72)); dias(mp(TB, TRSYN[:n]), RD, 5.5, .9)
    if p > .45: T("+ synthetic minority samples (training set only)", TB[0] + 8, TB[1] + TB[3] + 20, 14, RD, a=sg(p, .45, .55), w="normal")
    if p > .75:                                      # padlock icon + label
        a = sg(p, .75, .82); cx, cy = XB[0] + XB[2] / 2 - 40, XB[1] + XB[3] + 22
        ax.add_patch(Rectangle((cx - 7, cy - 4), 14, 11, fc=to_rgba(GR, a), ec="none", zorder=5))
        ax.add_patch(Arc((cx, cy - 4), 9, 12, theta1=180, theta2=360, ec=to_rgba(GR, a), lw=2 * PX, zorder=5))
        T("untouched", cx + 14, cy + 1, 15, GR, a=a)
    if p > .82:
        e = f(.82, .92)
        T("Apply ROS/SMOTE only to training data —", 480, 482, 24, AC, "center", e, "heavy")
        T("never oversample the test set.", 480, 514, 24, AC, "center", e, "heavy")


def s8(p):                                           # 9  Applications (vector icons)
    items = [("Credit-card fraud detection", "Rare fraudulent transactions"),
             ("Rare disease detection", "Few positive patient records"),
             ("Network intrusion / anomaly detection", "Attacks hide in normal traffic")]
    for i, (t1, t2) in enumerate(items):
        e = ez(sg(p, .1 + i * .2, .3 + i * .2)); x = 50 + i * 300; y = 150 + (1 - e) * 40
        ax.add_patch(FancyBboxPatch((x, y), 270, 250, boxstyle="round,pad=0,rounding_size=12",
                                    fc=to_rgba(BG, e), ec=to_rgba(LN, e), lw=1.5 * PX, zorder=1))
        cx, cy = x + 135, y + 80
        if i == 0:                                   # credit card
            ax.add_patch(FancyBboxPatch((cx - 38, cy - 24), 76, 48, boxstyle="round,pad=0,rounding_size=6", fc=to_rgba(BL, e), ec="none", zorder=3))
            rect(cx - 38, cy - 10, 76, 9, AC, e, 4); rect(cx - 30, cy + 10, 26, 6, PN, e, 4)
        elif i == 1:                                 # medical cross
            ax.add_patch(FancyBboxPatch((cx - 30, cy - 30), 60, 60, boxstyle="round,pad=0,rounding_size=8", fc=to_rgba(RD, e), ec="none", zorder=3))
            rect(cx - 5, cy - 20, 10, 40, PN, e, 4); rect(cx - 20, cy - 5, 40, 10, PN, e, 4)
        else:                                        # shield
            ax.add_patch(Polygon([(cx - 30, cy - 30), (cx + 30, cy - 30), (cx + 30, cy + 2), (cx, cy + 34), (cx - 30, cy + 2)],
                                 fc=to_rgba(GR, e), ec="none", zorder=3))
            ax.plot([cx - 13, cx - 3, cx + 15], [cy - 2, cy + 10, cy - 14], color=to_rgba(PN, e), lw=4 * PX, zorder=4)
        T(t1, x + 135, y + 160, 17 if i < 2 else 15, AC, "center", e); T(t2, x + 135, y + 200, 14, MUT, "center", e, "normal")


def s9(p):                                           # 10 Final summary
    R = (50, 100, 380, 260); f = f_(p); box(R, "Training data")
    mj(R); n = int(90 * f(.12, .5)); dias(mp(R, S1P[:n]), RD, 5, .85); mi(R)
    T(f"Minority: {50 + round(900 * f(.12, .5))}", R[0] + 8, R[1] + R[3] + 20, 15, RD)
    for i, s in enumerate(["Imbalanced Data", "↓  Oversampling", "Better Minority Representation"]):
        T(s, 480, 125 + i * 42, 20 if i == 1 else 23, MUT if i == 1 else AC, a=f(.2 + i * .08, .3 + i * .08), w="heavy")
    T("Random Oversampling: Duplicate existing points", 480, 290, 17, FG, a=f(.45, .52), w="normal")
    T("SMOTE: Generate synthetic points between neighbors", 480, 322, 17, FG, a=f(.5, .57), w="normal")
    e = f(.7, .82)
    T("“The goal is not simply higher accuracy —", 480, 432, 26, AC, "center", e, "heavy")
    T("it is learning the minority class better.”", 480, 470, 26, AC, "center", e, "heavy")


SCENES = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
NAMES = ["The Imbalanced Data Problem", "Why Accuracy Can Be Misleading", "Oversampling", "Random Oversampling (ROS)",
         "SMOTE — Synthetic Minority Over-sampling Technique", "ROS vs SMOTE", "Important Limitation",
         "Very Important Rule", "Applications", "Final Summary"]
DUR = [16, 14, 12, 16, 22, 24, 12, 14, 10, 12]      # seconds per scene
TOT = sum(DUR); CUM = np.cumsum([0] + DUR[:-1])


def draw(t):
    ax.cla(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); fig.patch.set_facecolor(PN)
    i = int(np.searchsorted(CUM, t, side="right") - 1)
    ax.add_patch(Circle((52, 42), 16, fc=AC, ec="none", zorder=3)); T(i + 1, 52, 42, 16, PN, "center", z=7)
    T(NAMES[i], 78, 42, 24, FG, w="heavy")
    SCENES[i](cl((t - CUM[i]) / DUR[i]))
    rect(0, H - 4, W * t / TOT, 4, AC, z=8)         # progress bar


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="oversampling_explainer.mp4")
    ap.add_argument("--preview", metavar="DIR", help="save one PNG per scene instead of rendering the video")
    a = ap.parse_args()
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
        for i in range(len(DUR)):
            draw(CUM[i] + DUR[i] * .93); fig.savefig(f"{a.preview}/scene{i + 1:02d}.png", dpi=100)
        print("bad synthetic points in scene 7:", sum(s["bad"] for s in S2)); return
    import imageio.v2 as imageio
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None)
    n = int(TOT * a.fps); t0 = time.time()
    for k in range(n):
        draw(min(k / a.fps, TOT - 1e-3)); w.append_data(grab())
        if k % (a.fps * 5) == 0: print(f"frame {k}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    w.close(); print("saved", a.out)


if __name__ == "__main__":
    main()