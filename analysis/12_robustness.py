"""Step 12 (V18, post-freeze): robustness of the route-1 embodiment effects.

A. Specification curve / multiverse (Simonsohn et al. 2020) over literature-grounded choices.
   Every specification is reported; none is selected. Joint inference: participant-label
   permutation (2,000) of the median Welch t across all specifications.
     P300 windows: 250-500 ms (this study), 320-380 ms (iCub, SCAN 2026), collapsed-localizer peak +-50 ms
     P300 ROIs   : Pz/POz (this study), Fz/F3/F4 (fronto-central, iCub 2026), P3/Pz/P4
     LPP windows : 400-800 ms (this study), 400-600 ms (Sollfrank 2021; CogSci 2024; Maier 2025),
                   collapsed-localizer peak +-50 ms
     LPP ROIs    : Pz/POz (this study), Cz/Pz/C3/C4 (centro-parietal, Maier 2025; Cz carries the
                   AFFEC audio confound, V1), P3/Pz/P4/POz
     x Miko data {pipeline, author}  x inclusion {3, 5, 8 trials}  x AFFEC lock {video, expression}
   Collapsed localizer: peak of the grand average pooled over both datasets with equal weight,
   so the window is blind to the physical-vs-virtual contrast.
B. Spatio-temporal cluster permutation test (Maris & Oostenveld 2007), 0-1000 ms x 16 channels,
   subject averages of the shared emotions in baseline-SD units, 2,000 permutations; no window
   choice at all.

Outputs results/robust_spec_curve.csv, results/robust_summary.csv, results/robust_clusters.csv,
        figures/fig6_spec_curve.png
"""
import itertools
import numpy as np
import pandas as pd
import mne
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from common import log, erp_view, eligible, shared_counts

R = cfg.RESULTS_DIR
rng = np.random.default_rng(cfg.SEED)
N_PERM = 2000

WINDOWS = {
    "P300": {"study 250-500": (0.25, 0.50), "iCub2026 320-380": (0.32, 0.38), "collapsed": (0.25, 0.60)},
    "LPP": {"study 400-800": (0.40, 0.80), "lit 400-600": (0.40, 0.60), "collapsed": (0.35, 0.90)},
}
ROIS = {
    "P300": {"study Pz/POz": ["Pz", "POz"], "fronto-central Fz/F3/F4": ["Fz", "F3", "F4"], "P3/Pz/P4": ["P3", "Pz", "P4"]},
    "LPP": {"study Pz/POz": ["Pz", "POz"], "centro-parietal Cz/Pz/C3/C4": ["Cz", "Pz", "C3", "C4"],
            "P3/Pz/P4/POz": ["P3", "Pz", "P4", "POz"]},
}


def subject_data(dataset, version):
    """Per participant: z-scaled ERP view (trials x ch x time) for the shared emotions + min count."""
    out = {}
    for f in sorted((cfg.DERIV_DIR / dataset).glob(f"sub-*_{version}-epo.fif")):
        ep = mne.read_epochs(f, preload=True, verbose=False)
        if not eligible(ep, min(cfg.MIN_TRIALS_SENS)):
            continue
        ep = ep[ep.metadata.emotion.isin(cfg.SHARED_EMOTIONS).values]
        out[f.name.split("_")[0][4:]] = (ep, min(shared_counts(ep).values()))
    return out


def expr_locked(subjects):
    """AFFEC long epochs re-locked to expression onset (same routine as step 05)."""
    import importlib
    E = importlib.import_module("05_erp_features")
    stim = pd.read_csv(R / "affec_stim_features.csv")
    out = {}
    for s, (ep, n) in subjects.items():
        lep = mne.read_epochs(cfg.DERIV_DIR / "affec" / f"sub-{s}_long-epo.fif", preload=True, verbose=False)
        lep = lep[lep.metadata.emotion.isin(cfg.SHARED_EMOTIONS).values]
        v = E.relock_to_expression(lep, stim)
        if v is not None and len(v):
            out[s] = (v, min(shared_counts(v).values()), True)
    return out


def zview(ep, already_view=False):
    v = ep if already_view else erp_view(ep)
    d = v.get_data(picks=cfg.COMMON_CHANNELS)
    t = v.times
    sd = d[:, :, (t >= cfg.BASELINE[0]) & (t <= cfg.BASELINE[1])].transpose(1, 0, 2).reshape(d.shape[1], -1).std(1)
    z = d / sd[None, :, None]
    # subject mean over the three emotions (equal weight per emotion)
    emo = v.metadata.emotion.values
    return np.mean([z[emo == e].mean(0) for e in cfg.SHARED_EMOTIONS], axis=0), t


def build(miko_version, affec_lock):
    mk = {s: zview(ep) + (n,) for s, (ep, n) in subject_data("miko", miko_version).items()}
    af = subject_data("affec", "pipeline")
    if affec_lock == "expression":
        af = {s: zview(v, True) + (n,) for s, (v, n, _) in expr_locked(af).items()}
    else:
        af = {s: zview(ep) + (n,) for s, (ep, n) in af.items()}
    return mk, af


def collapsed_window(mk, af, chans, broad, times):
    idx = [cfg.COMMON_CHANNELS.index(c) for c in chans]
    ga = 0.5 * (np.mean([z[idx].mean(0) for z, _, _ in mk.values()], 0) +
                np.mean([z[idx].mean(0) for z, _, _ in af.values()], 0))
    m = (times >= broad[0]) & (times <= broad[1])
    pk = times[m][np.argmax(ga[m])]
    return (pk - 0.05, pk + 0.05)


def main():
    rows, mats = [], {}
    for mv, lock in itertools.product(("pipeline", "author"), ("video", "expression")):
        mk, af = build(mv, lock)
        times = next(iter(mk.values()))[1]
        for comp in ("P300", "LPP"):
            for (wname, win), (rname, chans), k in itertools.product(WINDOWS[comp].items(), ROIS[comp].items(),
                                                                      (3, 5, 8)):
                w = collapsed_window(mk, af, chans, win, times) if wname == "collapsed" else win
                idx = [cfg.COMMON_CHANNELS.index(c) for c in chans]
                tm = (times >= w[0]) & (times <= w[1])
                xs = {("miko", sid): z[idx][:, tm].mean() for sid, (z, _, n) in mk.items() if n >= k}
                ys = {("affec", sid): z[idx][:, tm].mean() for sid, (z, _, n) in af.items() if n >= k}
                x, y = np.array(list(xs.values())), np.array(list(ys.values()))
                spec = dict(component=comp, window=wname, win_s=f"{w[0]:.3f}-{w[1]:.3f}", roi=rname,
                            miko=mv, affec_lock=lock, min_trials=k, n_phys=len(x), n_virt=len(y))
                t, p = stats.ttest_ind(x, y, equal_var=False)
                g = (x.mean() - y.mean()) / np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
                rows.append({**spec, "diff_phys_minus_virt": x.mean() - y.mean(), "hedges_g": g, "t": t, "p": p})
                mats[len(rows) - 1] = {**xs, **ys}
        log("multiverse block done:", mv, lock)
    spec = pd.DataFrame(rows)
    spec.to_csv(R / "robust_spec_curve.csv", index=False, encoding="utf-8-sig")

    # joint null: ONE shuffle of the participant labels per permutation, applied to every
    # specification (specifications share participants, so they must be permuted together)
    units = sorted({u for m in mats.values() for u in m})
    is_phys = np.array([u[0] == "miko" for u in units])
    pos = {u: i for i, u in enumerate(units)}
    packed = {i: (np.array([pos[u] for u in m]), np.array(list(m.values()))) for i, m in mats.items()}
    perms = [rng.permutation(is_phys) for _ in range(N_PERM)]

    def welch_t(v, lab):
        a, b = v[lab], v[~lab]
        return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    summ = []
    for comp, d in spec.groupby("component"):
        ids = d.index.values
        obs = np.median(d.t)
        null = np.array([np.median([welch_t(packed[i][1], L[packed[i][0]]) for i in ids]) for L in perms])
        summ.append(dict(component=comp, n_specs=len(d), median_t=obs, median_g=d.hedges_g.median(),
                         frac_p05=(d.p < .05).mean(), frac_p05_positive=((d.p < .05) & (d.t > 0)).mean(),
                         frac_p05_negative=((d.p < .05) & (d.t < 0)).mean(),
                         p_joint_perm=(1 + np.sum(np.abs(null) >= abs(obs))) / (N_PERM + 1)))
    summ = pd.DataFrame(summ)
    summ.to_csv(R / "robust_summary.csv", index=False, encoding="utf-8-sig")
    log(summ.round(4).to_string())

    # --- B: cluster permutation over time x channels, for both AFFEC time-locking events ----
    info = mne.create_info(cfg.COMMON_CHANNELS, cfg.SFREQ, "eeg")
    info.set_montage(cfg.MONTAGE, on_missing="ignore")
    adj, _ = mne.channels.find_ch_adjacency(info, "eeg")
    cl = []
    for lock in ("video", "expression"):
        mk, af = build("pipeline", lock)
        times = next(iter(mk.values()))[1]
        tm = (times >= 0) & (times <= 1.0)
        X = [np.stack([z[:, tm].T for z, _, n in d.values() if n >= cfg.MIN_TRIALS_SHARED]) for d in (mk, af)]
        # signed Welch t, two-tailed: positive and negative differences form separate clusters
        tcrit = stats.t.ppf(0.975, len(X[0]) + len(X[1]) - 2)
        tobs, clusters, pv, _ = mne.stats.spatio_temporal_cluster_test(
            X, adjacency=adj, n_permutations=N_PERM, tail=0, seed=cfg.SEED, out_type="mask", verbose=False,
            stat_fun=lambda a, b: stats.ttest_ind(a, b, equal_var=False)[0], threshold=tcrit)
        tt = times[tm]
        np.savez_compressed(cfg.DERIV_DIR / f"robust_cluster_{lock}.npz", t_obs=tobs, times=tt,
                            masks=np.array(clusters) if clusters else np.zeros((0,) + tobs.shape, bool),
                            p=np.array(pv), channels=np.array(cfg.COMMON_CHANNELS))
        for c, p in zip(clusters, pv):
            ti, ci = np.where(c)
            cl.append(dict(affec_lock=lock, p=p, t_start=tt[ti.min()], t_end=tt[ti.max()],
                           channels=",".join(sorted({cfg.COMMON_CHANNELS[j] for j in ci})),
                           direction="physical > virtual" if tobs[c].mean() > 0 else "physical < virtual",
                           size=int(c.sum()), n_phys=len(X[0]), n_virt=len(X[1])))
    cl = pd.DataFrame(cl).sort_values(["affec_lock", "p"]) if cl else pd.DataFrame([dict(p=np.nan, note="no clusters")])
    cl.to_csv(R / "robust_clusters.csv", index=False, encoding="utf-8-sig")
    log(cl.groupby("affec_lock").head(4).round(4).to_string())

    # --- figure: specification curve -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, (comp, d) in zip(ax, spec.groupby("component")):
        d = d.sort_values("hedges_g").reset_index(drop=True)
        col = np.where(d.p < .05, "#c0392b", "0.6")
        a.scatter(range(len(d)), d.hedges_g, c=col, s=8)
        a.axhline(0, color="k", lw=.6)
        s = summ.set_index("component").loc[comp]
        a.set(title=f"{comp}: {len(d)} specifications, joint p = {s.p_joint_perm:.3f}",
              xlabel="specification (sorted)", ylabel="Hedges g (physical − virtual)")
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "fig6_spec_curve.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
