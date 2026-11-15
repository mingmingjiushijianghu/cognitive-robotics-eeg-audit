"""Step 8 (design §4): route 2 - event-related dwPLI networks and topology.

dwPLI (debiased weighted phase-lag index, Vinck et al. 2011) from multitaper cross-spectra
(DPSS, 0.5 s windows, 2 Hz half-bandwidth) on the broadband epochs:
  base = -0.5..0 s; post = mean of 0..0.5 s and 0.5..1.0 s (equal-length windows, so the
  estimator has the same number of samples in both). Delta = post - base.
Trial counts are equalised: per participant x emotion, CONN_N trials are drawn at random,
CONN_REPEATS times; the per-trial cross-spectra are computed once, so every repeat only sums
a subset. Cells with fewer than CONN_N trials are dropped.

Network level: module means (frontal, parietal, temporal; aux nodes enter the whole-graph
metrics only), proportional thresholds 10-50 % -> AUC of global efficiency, mean clustering
and modularity (bctpy), each as post - base.
Statistics: LMM metric ~ embodiment * emotion + (1 | subject) per band (FDR across the
confirmatory family); NBS (bctpy, 5,000 permutations) on Delta matrices, physical vs virtual.
Sensitivity: Miko author-cleaned data (beta-band muscle residue, §3.1).

Outputs derivatives/conn/<dataset>_<version>_<subject>.npz,
        results/conn_{module,topology}.csv, results/conn_stats.csv, results/conn_nbs.csv
"""
import numpy as np
import pandas as pd
import mne
import bct
import statsmodels.formula.api as smf
from scipy import stats
from scipy.signal.windows import dpss
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import config as C
from common import log

OUT = C.DERIV_DIR / "conn"
OUT.mkdir(parents=True, exist_ok=True)
CONN_N = 10
WIN = 0.5
NW = 1.0                                     # time-half-bandwidth -> 2 Hz half-bandwidth at 0.5 s
# validated against mne_connectivity wpli2_debiased (mt_bandwidth=4): r = .9995 on Miko P1
CH = C.COMMON_CHANNELS
IU = np.triu_indices(len(CH), 1)


def tapered_spectra(x, sfreq):
    """x: trials x ch x samples -> trials x tapers x ch x freqs (complex)."""
    n = x.shape[-1]
    tapers = dpss(n, NW, int(2 * NW - 1))
    xt = x[:, None, :, :] * tapers[None, :, None, :]
    return np.fft.rfft(xt, axis=-1), np.fft.rfftfreq(n, 1 / sfreq)


def imag_cross(x, sfreq):
    """Per-trial imaginary cross-spectrum averaged over tapers, per band: dict band -> trials x f x ch x ch."""
    X, f = tapered_spectra(x, sfreq)
    out = {}
    for b, (lo, hi) in C.BANDS.items():
        sel = (f >= lo) & (f <= hi)
        Xb = X[..., sel]                                               # tr x tap x ch x f
        S = np.einsum("tkif,tkjf->tfij", Xb, np.conj(Xb)) / Xb.shape[1]
        out[b] = S.imag
    return out


def dwpli(im):
    """im: trials x f x ch x ch -> ch x ch (debiased wPLI, averaged over frequencies)."""
    s, a, q = im.sum(0), np.abs(im).sum(0), (im ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (s ** 2 - q) / (a ** 2 - q)
    return np.nan_to_num(w).mean(0)


def windows(ep):
    sf = ep.info["sfreq"]
    n = int(round(WIN * sf))
    t = ep.times
    i0 = lambda tt: int(np.argmin(np.abs(t - tt)))
    d = ep.get_data(picks=CH)
    return {"base": d[..., i0(-0.5):i0(-0.5) + n], "post1": d[..., i0(0.0):i0(0.0) + n],
            "post2": d[..., i0(0.5):i0(0.5) + n]}, sf


def subject_matrices(f, dataset, version, rng_seed):
    subject = f.name.split("_")[0].replace("sub-", "")
    ep = mne.read_epochs(f, preload=True, verbose=False)
    ep = ep[ep.metadata.emotion.isin(C.SHARED_EMOTIONS).values]
    rng = np.random.default_rng(rng_seed)
    res = {}
    for emo in C.SHARED_EMOTIONS:
        e = ep[(ep.metadata.emotion == emo).values]
        if len(e) < CONN_N:
            continue
        w, sf = windows(e)
        ims = {k: imag_cross(v, sf) for k, v in w.items()}
        for b in C.BANDS:
            acc = {"base": 0, "post": 0}
            for _ in range(C.CONN_REPEATS):
                pick = rng.choice(len(e), CONN_N, replace=False)
                acc["base"] = acc["base"] + dwpli(ims["base"][b][pick])
                acc["post"] = acc["post"] + 0.5 * (dwpli(ims["post1"][b][pick]) + dwpli(ims["post2"][b][pick]))
            res[(emo, b)] = {k: v / C.CONN_REPEATS for k, v in acc.items()}
    np.savez_compressed(OUT / f"{dataset}_{version}_{subject}.npz",
                        **{f"{e}|{b}|{k}": m for (e, b), d in res.items() for k, m in d.items()})
    return dataset, version, subject, res


def module_means(m):
    idx = {k: [CH.index(c) for c in v] for k, v in C.MODULES.items()}
    def blk(a, b):
        sub = m[np.ix_(idx[a], idx[b])]
        return sub[np.triu_indices(len(idx[a]), 1)].mean() if a == b else sub.mean()
    return {"F-F": blk("frontal", "frontal"), "P-P": blk("parietal", "parietal"),
            "F-P": blk("frontal", "parietal"), "F-T": blk("frontal", "temporal"),
            "P-T": blk("parietal", "temporal")}


def topology(m):
    w = np.clip(m, 0, None)
    np.fill_diagonal(w, 0)
    out = {"geff": [], "clust": [], "Q": []}
    for d in C.DENSITIES:
        t = bct.threshold_proportional(w, d)
        b = (t > 0).astype(float)
        out["geff"].append(bct.efficiency_bin(b))
        out["clust"].append(np.mean(bct.clustering_coef_bu(b)))
        out["Q"].append(bct.modularity_und(t, seed=C.SEED)[1] if t.any() else 0.0)
    x = np.array(C.DENSITIES)
    return {k: float(np.trapezoid(v, x)) for k, v in out.items()}


def lmm_tests(df, value_cols, tag):
    rows = []
    for (band, metric), d in df.melt(id_vars=["dataset", "subject", "emotion", "band"],
                                      value_vars=value_cols, var_name="metric").groupby(["band", "metric"]):
        d = d.assign(embodiment=np.where(d.dataset == "miko", "physical", "virtual"))
        fit = lambda fx: smf.mixedlm(f"value ~ {fx}", d, groups="subject").fit(reml=False)
        full, noint, noemb = fit("embodiment * emotion"), fit("embodiment + emotion"), fit("emotion")
        for eff, a, b in (("embodiment", noint, noemb), ("embodiment x emotion", full, noint)):
            s = 2 * (a.llf - b.llf); k = len(a.fe_params) - len(b.fe_params)
            beta = noint.fe_params.get("embodiment[T.virtual]", np.nan) if eff == "embodiment" else np.nan
            rows.append(dict(analysis=tag, band=band, metric=metric, effect=eff, chi2=s, df=k,
                             p=stats.chi2.sf(s, k), beta_virtual_minus_physical=beta))
    return pd.DataFrame(rows)


def main():
    jobs = [("miko", v, f) for v in ("pipeline", "author") for f in sorted((C.DERIV_DIR / "miko").glob(f"sub-*_{v}-epo.fif"))]
    jobs += [("affec", "pipeline", f) for f in sorted((C.DERIV_DIR / "affec").glob("sub-*_pipeline-epo.fif"))]
    log(len(jobs), "epoch files")
    res = Parallel(n_jobs=8)(delayed(subject_matrices)(f, d, v, C.SEED + i) for i, (d, v, f) in enumerate(jobs))

    mod, top, deltas = [], [], {}
    for dataset, version, subject, r in res:
        for (emo, band), d in r.items():
            delta = d["post"] - d["base"]
            key = dict(dataset=dataset, version=version, subject=subject, emotion=emo, band=band)
            mod.append({**key, **module_means(delta)})
            tp, tb = topology(d["post"]), topology(d["base"])
            top.append({**key, **{k: tp[k] - tb[k] for k in tp}})
            deltas.setdefault((dataset, version, band, subject), []).append(delta)
    mod, top = pd.DataFrame(mod), pd.DataFrame(top)
    mod.to_csv(C.RESULTS_DIR / "conn_module.csv", index=False, encoding="utf-8-sig")
    top.to_csv(C.RESULTS_DIR / "conn_topology.csv", index=False, encoding="utf-8-sig")

    tests = []
    for tag, mv in (("primary", "pipeline"), ("miko_author", "author")):
        sel = lambda df: df[((df.dataset == "miko") & (df.version == mv)) | ((df.dataset == "affec") & (df.version == "pipeline"))]
        tests.append(lmm_tests(sel(mod), ["F-F", "P-P", "F-P", "F-T", "P-T"], tag))
        tests.append(lmm_tests(sel(top), ["geff", "clust", "Q"], tag))
    tests = pd.concat(tests, ignore_index=True)
    prim = tests.analysis == "primary"
    tests.loc[prim, "p_fdr"] = multipletests(tests.loc[prim, "p"], method="fdr_bh")[1]
    tests.to_csv(C.RESULTS_DIR / "conn_stats.csv", index=False, encoding="utf-8-sig")

    nbs = []
    for band in C.BANDS:
        x = [np.mean(v, 0) for (d, ver, b, s), v in deltas.items() if d == "miko" and ver == "pipeline" and b == band]
        y = [np.mean(v, 0) for (d, ver, b, s), v in deltas.items() if d == "affec" and b == band]
        if len(x) < 3 or len(y) < 3:
            continue
        for thr in (2.5, 3.0, 3.5):
            pval, adj, null = bct.nbs_bct(np.stack(x, -1), np.stack(y, -1), thresh=thr, k=C.N_PERM,
                                          tail="both", paired=False, seed=C.SEED)
            for ci, p in enumerate(np.atleast_1d(pval)):
                comp = np.argwhere(np.triu(adj == ci + 1))
                nbs.append(dict(band=band, t_thresh=thr, component=ci + 1, p=float(p), n_edges=len(comp),
                                edges=";".join(f"{CH[i]}-{CH[j]}" for i, j in comp)))
            if not np.atleast_1d(pval).size:
                nbs.append(dict(band=band, t_thresh=thr, component=0, p=np.nan, n_edges=0, edges=""))
    pd.DataFrame(nbs).to_csv(C.RESULTS_DIR / "conn_nbs.csv", index=False, encoding="utf-8-sig")
    log(tests[prim].sort_values("p").head(12).round(4).to_string())


if __name__ == "__main__":
    main()
