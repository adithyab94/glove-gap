"""Aggregate eval/out/*.csv -> report/results.csv + report/results.md (condition x model x metric).

Unit of analysis = sequence: each metric is averaged within a sequence (both pair cams pooled),
then the mean over sequences is reported with a bootstrap 95% CI over sequences (n=2000).
Deltas vs G0 use a paired bootstrap (same resampled sequences for both conditions).
Main set excludes analysis.exclude_main (appendix shows with/without).
Usage: summary.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import ROOT, CFG, sequences, conditions

OUT = ROOT / "eval/out"; REP = ROOT / "report"; REP.mkdir(exist_ok=True)
B = CFG["analysis"]["bootstrap"]["n"]
rng = np.random.default_rng(CFG["seed"])
MAIN = sequences(); ALL = sequences(include_excluded=True)
CONDS = conditions()
FRAME_METRICS = ["epe2d_px", "pck02", "mpjpe_rel_mm", "pa_mpjpe_mm", "mpjpe_abs_mm", "root_err_mm",
                 "wrist_depth_err_mm", "aperture_err_mm"]
TWO_VIEW = ["tri_rate", "mpjpe_abs_mm", "mpjpe_rel_mm", "reproj_px", "jitter_mm_f2", "aperture_err_mm"]


def boot(seq_vals, seqs):
    """seq_vals: Series indexed by seq. -> mean, lo, hi over sequences."""
    v = seq_vals.reindex(seqs).dropna().values
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(v, (B, len(v))).mean(1)
    return v.mean(), *np.percentile(bs, [2.5, 97.5])


def boot_delta(a, b, seqs):
    """paired: mean(b - a) over sequences present in both."""
    d = (b.reindex(seqs) - a.reindex(seqs)).dropna().values
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(d, (B, len(d))).mean(1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def seq_table(pf, seqs):
    """per (model, cond, seq): detection rate + metric means on detected frames."""
    pf = pf[pf.seq.isin(seqs)]
    det = pf.groupby(["model", "cond", "seq"]).detected.mean().rename("det_rate")
    m = pf[pf.detected == 1].groupby(["model", "cond", "seq"])[FRAME_METRICS].mean()
    return pd.concat([det, m], axis=1)


def common_subset(pf):
    """frames (cond, seq, cam, frame) that MediaPipe AND WiLoR detected (HaMeR == WiLoR boxes)."""
    key = ["cond", "seq", "cam", "frame"]
    d = pf.pivot_table(index=key, columns="model", values="detected")
    ok = d[(d.get("mediapipe", 0) == 1) & (d.get("wilor", 0) == 1)].index
    return pf.set_index(key).loc[lambda x: x.index.isin(ok)].reset_index()


def rows_for(tab, seqs, scope, metrics):
    out = []
    for (model, cond), g in tab.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        for met in metrics:
            if met not in g or g[met].isna().all():
                continue
            mean, lo, hi = boot(g[met], seqs)
            r = {"scope": scope, "model": model, "cond": cond, "metric": met,
                 "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_seq": int(g[met].reindex(seqs).notna().sum())}
            if cond != "G0" and (model, "G0") in tab.index.droplevel(2):
                base = tab.loc[(model, "G0")][met]
                r["delta_vs_G0"], r["d_lo"], r["d_hi"] = boot_delta(base, g[met], seqs)
            out.append(r)
    return out


if __name__ == "__main__":
    pf = pd.read_csv(OUT / "per_frame.csv")
    tv = pd.read_csv(OUT / "per_seq_2view.csv").set_index(["model", "cond", "seq"])
    rows = []
    rows += rows_for(seq_table(pf, MAIN), MAIN, "per_view_detected", ["det_rate"] + FRAME_METRICS)
    rows += rows_for(seq_table(common_subset(pf), MAIN), MAIN, "per_view_common_subset", FRAME_METRICS)
    rows += rows_for(tv, MAIN, "two_view", TWO_VIEW)
    res = pd.DataFrame(rows)
    res.to_csv(REP / "results.csv", index=False, float_format="%.4g")

    # ---- per-sequence table: skin share, flags, outliers
    fr = pd.read_csv(ROOT / "data/cache/frames.csv", dtype={"cam": str})  # serials, not ints
    fr = fr[(fr.condition == "G0") & fr.cam.isin(CFG["dexycb"]["pair"])]
    skin = fr.groupby("seq").skin_ring.mean()
    st = seq_table(pf, ALL)
    per_seq = pd.DataFrame({"skin_share": skin})
    meta = pd.read_csv(ROOT / "data/meta.csv").drop_duplicates("seq").set_index("seq")
    per_seq["task"] = meta.task; per_seq["side"] = meta.side
    for model in ["mediapipe", "wilor"]:
        for cond in CONDS:
            if (model, cond) in st.index.droplevel(2):
                per_seq[f"{model}_{cond}_det"] = st.loc[(model, cond)].det_rate
    for cond in CONDS:
        per_seq[f"wilor_{cond}_pa"] = st.loc[("wilor", cond)].pa_mpjpe_mm
    per_seq["flag"] = [CFG["analysis"]["flag"].get(s, "") + (" EXCLUDED(main)" if s in CFG["analysis"]["exclude_main"] else "")
                       for s in per_seq.index]
    # outlier = WiLoR PA-MPJPE above Q3 + 1.5 IQR (main set) in any condition
    out_flags = {}
    for cond in CONDS:
        v = per_seq.loc[MAIN, f"wilor_{cond}_pa"]   # WiLoR runs on every condition
        q1, q3 = v.quantile([.25, .75]); lim = q3 + 1.5 * (q3 - q1)
        for s in v[v > lim].index:
            out_flags.setdefault(s, []).append(cond)
    per_seq["pa_outlier_in"] = [",".join(out_flags.get(s, [])) for s in per_seq.index]
    per_seq.to_csv(REP / "per_sequence.csv", float_format="%.4g")

    # ---- skin share vs degradation (main set)
    corr = []
    for model, met, lab in [("mediapipe", "det", "det drop"), ("wilor", "det", "det drop"), ("wilor", "pa", "PA-MPJPE increase")]:
        for cond in ["G1", "G2", "G3a"]:
            a, b = per_seq.loc[MAIN, f"{model}_G0_{met}"], per_seq.loc[MAIN, f"{model}_{cond}_{met}"]
            d = (a - b) if met == "det" else (b - a)
            rho, p = spearmanr(per_seq.loc[MAIN, "skin_share"], d, nan_policy="omit")
            corr.append({"model": model, "cond": cond, "quantity": lab, "spearman_rho": rho, "p": p})
    pd.DataFrame(corr).to_csv(REP / "skin_share_corr.csv", index=False, float_format="%.3g")

    # ---- appendix: with vs without excluded/flagged sequences (WiLoR PA-MPJPE, MP detection)
    app = []
    tab_all = seq_table(pf, ALL)
    for label, seqs in [("main (n=%d)" % len(MAIN), MAIN), ("with excluded (n=%d)" % len(ALL), ALL)] + \
            [(f"main minus flagged {s}", [x for x in MAIN if x != s]) for s in CFG["analysis"]["flag"]]:
        for model, met in [("wilor", "pa_mpjpe_mm"), ("mediapipe", "det_rate"), ("wilor", "det_rate")]:
            for cond in ["G0", "G2", "G3a"]:
                m, lo, hi = boot(tab_all.loc[(model, cond)][met], seqs)
                app.append({"set": label, "model": model, "metric": met, "cond": cond, "mean": m, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(app).to_csv(REP / "appendix_exclusions.csv", index=False, float_format="%.4g")

    # ---- markdown headline table
    def cell(model, cond, met, scope="per_view_detected", pct=False):
        r = res[(res.scope == scope) & (res.model == model) & (res.cond == cond) & (res.metric == met)]
        if r.empty:
            return "–"
        r = r.iloc[0]; f = (lambda x: f"{100 * x:.0f}") if pct else (lambda x: f"{x:.1f}")
        n = f" (n={int(r.n_seq)})" if r.n_seq < len(MAIN) else ""
        return f"{f(r['mean'])} [{f(r.ci_lo)}–{f(r.ci_hi)}]{n}"
    lines = ["| Condition | MP det % | WiLoR det % | MP PCK@0.2 % | WiLoR PA-MPJPE mm | HaMeR PA-MPJPE mm | WiLoR MPJPE(rel) mm | 2-view MPJPE MP / WiLoR / HaMeR mm |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CONDS:
        lines.append(f"| {c} | {cell('mediapipe', c, 'det_rate', pct=True)} | {cell('wilor', c, 'det_rate', pct=True)} | "
                     f"{cell('mediapipe', c, 'pck02', pct=True)} | {cell('wilor', c, 'pa_mpjpe_mm')} | {cell('hamer', c, 'pa_mpjpe_mm')} | "
                     f"{cell('wilor', c, 'mpjpe_rel_mm')} | {cell('mediapipe', c, 'mpjpe_abs_mm', 'two_view')} / "
                     f"{cell('wilor', c, 'mpjpe_abs_mm', 'two_view')} / {cell('hamer', c, 'mpjpe_abs_mm', 'two_view')} |")
    (REP / "results.md").write_text(
        f"Mean over {len(MAIN)} sequences [bootstrap 95% CI over sequences]. Errors on detected frames.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
