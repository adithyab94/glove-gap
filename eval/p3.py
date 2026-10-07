"""P3: interventions + trainability gate, from cached predictions (no new inference).

Pipelines per (model, condition, sequence), WiLoR and HaMeR:
  mono       single view (each pair cam scored separately, then averaged): absolute 3D from
             root-relative MANO joints + translation re-solved with true K; aperture from kp3d_rel
  2view      DLT triangulation of the two pair cams' 2D keypoints
  2view_oe   2view after a One-Euro filter on each view's 2D keypoint tracks (params tuned on G0)
  hybrid     wrist position triangulated from both views + monocular articulation (each view's
             root-relative joints rotated into world, averaged over views)
  8cam       from eval/out/upper_bound.csv (WiLoR; G0/G2/G3a/G3b_k4; every 2nd frame)
Gate: frame valid if triangulated from both views with mean reprojection error < tau
(tau = 95th percentile of WiLoR G0 frames); episode passes if >= X% of GT-visible frames valid.
Outputs: eval/out/p3_pipelines.csv, p3_events.csv, p3_gate.csv, report/recovery.csv/.md
Usage: p3.py
"""
import csv, itertools
import numpy as np, pandas as pd
from common import (ROOT, CFG, PAIR, FPS, gt, load_pred, match, procrustes, solve_translation, world_from_cam,
                    triangulate, proj_matrix, reproj_err, intrinsics, sequences, conditions)
from metrics import gt_world, onset, TIP_T, TIP_I

OUT = ROOT / "eval/out"; REP = ROOT / "report"
SEQS = sequences(); CONDS = conditions()
MODELS = ["wilor", "hamer"]
TOL = round(CFG["events"]["tolerance_s"] * FPS)
GT_EV = {r["seq"]: int(r["frame"]) for r in csv.DictReader(open(ROOT / "data/events.csv"))}
rng = np.random.default_rng(CFG["seed"])


# ------------------------------------------------------------------ One-Euro filter
class OneEuro:
    """Casiez et al. 2012, applied per coordinate; fps-normalised (t in frames / FPS)."""

    def __init__(self, mincutoff, beta, dcutoff=1.0):
        self.mc, self.b, self.dc = mincutoff, beta, dcutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cut):
        tau = 1.0 / (2 * np.pi * cut); te = 1.0 / FPS
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x); return x
        dx = (x - self.x) * FPS
        self.dx = self.dx + self._a(self.dc) * (dx - self.dx)
        cut = self.mc + self.b * np.abs(self.dx)
        self.x = self.x + self._a(cut) * (x - self.x)
        return self.x


def filter_track(k2_by_f, mc, beta, max_gap=2):
    """k2_by_f: {frame: (21,2)} -> filtered dict; filter resets across gaps > max_gap frames."""
    out, f_prev, filt = {}, None, None
    for f in sorted(k2_by_f):
        if filt is None or f - f_prev > max_gap:
            filt = OneEuro(mc, beta)
        out[f] = filt(np.asarray(k2_by_f[f], float)); f_prev = f
    return out


# ------------------------------------------------------------------ data
def matched(model, cond, seq, cam):
    """{frame: matched hand dict} for frames with valid GT joints in this view."""
    p = load_pred(model, cond, seq, cam); out = {}
    for f, hands in p.items():
        _, j2, _ = gt(seq, cam, f)
        if (j2 < 0).any():
            continue
        h = match(hands, j2)
        if h is not None:
            out[f] = h
    return out


def accel(X):
    a = [np.linalg.norm(X[f + 1] - 2 * X[f] + X[f - 1], axis=1).mean() for f in X if f - 1 in X and f + 1 in X]
    return 1000 * np.mean(a) if a else np.nan


def run_mono(model, cond, seq):
    rows, traces = [], {}
    for cam in PAIR:
        m = matched(model, cond, seq, cam); K = intrinsics(cam)
        A, ap, errs_abs, errs_rel, zb = {}, {}, [], [], []
        frames = sorted(load_pred(model, cond, seq, cam))
        for f, h in m.items():
            j3, _, vis = gt(seq, cam, f)
            rel = np.asarray(h["kp3d_rel"]); a = rel + solve_translation(rel, np.asarray(h["kp2d"]), K)
            A[f] = a; ap[f] = 1000 * np.linalg.norm(rel[TIP_T] - rel[TIP_I])
            if vis:
                errs_abs.append(1000 * np.linalg.norm(a - j3, axis=1).mean())
                errs_rel.append(1000 * np.linalg.norm(rel - (j3 - j3[0]), axis=1).mean())
                zb.append(1000 * (a[0, 2] - j3[0, 2]))   # + = predicted farther than GT
        gap = {f: 1000 * np.linalg.norm(gt(seq, cam, f)[0][TIP_T] - gt(seq, cam, f)[0][TIP_I]) for f in frames}
        traces[cam] = (ap, gap, frames)
        rows.append({"mpjpe_abs_mm": np.mean(errs_abs) if errs_abs else np.nan,
                     "mpjpe_rel_mm": np.mean(errs_rel) if errs_rel else np.nan, "jitter_mm_f2": accel(A),
                     "root_z_bias_mm": np.mean(zb) if zb else np.nan})
    return {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}, traces


def run_2view(model, cond, seq, oe=None):
    m = {cam: matched(model, cond, seq, cam) for cam in PAIR}
    k2 = {cam: {f: h["kp2d"] for f, h in m[cam].items()} for cam in PAIR}
    if oe is not None:
        k2 = {cam: filter_track(k2[cam], *oe) for cam in PAIR}
    Ps = {cam: proj_matrix(seq, cam) for cam in PAIR}
    frames = sorted(load_pred(model, cond, seq, PAIR[0]))
    X, re, abs_e, rel_e = {}, {}, [], []
    for f in frames:
        if not all(f in k2[c] for c in PAIR):
            continue
        X[f] = triangulate([np.asarray(k2[c][f]) for c in PAIR], [Ps[c] for c in PAIR])
        re[f] = np.mean([reproj_err(X[f], np.asarray(k2[c][f]), Ps[c]).mean() for c in PAIR])
    vis_frames = [f for f in frames if any(gt(seq, c, f)[2] for c in PAIR)]
    G = {f: gt_world(seq, PAIR[0], f) for f in frames}
    for f in X:
        if f in vis_frames:
            abs_e.append(1000 * np.linalg.norm(X[f] - G[f], axis=1).mean())
            rel_e.append(1000 * np.linalg.norm((X[f] - X[f][0]) - (G[f] - G[f][0]), axis=1).mean())
    ap = {f: 1000 * np.linalg.norm(X[f][TIP_T] - X[f][TIP_I]) for f in X}
    gap = {f: 1000 * np.linalg.norm(G[f][TIP_T] - G[f][TIP_I]) for f in frames}
    res = {"mpjpe_abs_mm": np.mean(abs_e) if abs_e else np.nan, "mpjpe_rel_mm": np.mean(rel_e) if rel_e else np.nan,
           "jitter_mm_f2": accel(X)}
    return res, (ap, gap, frames), {f: re[f] for f in X}, vis_frames


def run_hybrid(model, cond, seq):
    m = {cam: matched(model, cond, seq, cam) for cam in PAIR}
    Ps = {cam: proj_matrix(seq, cam) for cam in PAIR}
    Rs = {cam: world_from_cam(seq, cam)[:3, :3] for cam in PAIR}
    frames = sorted(load_pred(model, cond, seq, PAIR[0]))
    vis_frames = {f for f in frames if any(gt(seq, c, f)[2] for c in PAIR)}
    X, abs_e, rel_e = {}, [], []
    for f in frames:
        if not all(f in m[c] for c in PAIR):
            continue
        wrist = triangulate([np.asarray(m[c][f]["kp2d"]) for c in PAIR], [Ps[c] for c in PAIR])[0]
        rel = np.mean([(Rs[c] @ np.asarray(m[c][f]["kp3d_rel"]).T).T for c in PAIR], axis=0)
        X[f] = wrist + rel
        if f in vis_frames:
            G = gt_world(seq, PAIR[0], f)
            abs_e.append(1000 * np.linalg.norm(X[f] - G, axis=1).mean())
            rel_e.append(1000 * np.linalg.norm(rel - (G - G[0]), axis=1).mean())
    gap = {f: 1000 * np.linalg.norm(gt_world(seq, PAIR[0], f)[TIP_T] - gt_world(seq, PAIR[0], f)[TIP_I]) for f in frames}
    ap = {f: 1000 * np.linalg.norm(X[f][TIP_T] - X[f][TIP_I]) for f in X}
    return ({"mpjpe_abs_mm": np.mean(abs_e) if abs_e else np.nan, "mpjpe_rel_mm": np.mean(rel_e) if rel_e else np.nan,
             "jitter_mm_f2": accel(X)}, (ap, gap, frames))


# ------------------------------------------------------------------ events (generic)
def event_scores(traces_by_cond):
    """traces_by_cond: {cond: {seq: [trace, ...]}} (several traces = several views, scored
    separately and averaged). rho + lag tuned on G0, then applied to every condition."""
    def f1_of(cond, rho, lag):
        f1s = []
        for tr_list in zip(*[traces_by_cond[cond][s] for s in SEQS]):   # one view index at a time
            tp = npred = 0
            for s, (ap, _, fr) in zip(SEQS, tr_list):
                p = onset(ap, fr, rho)
                if p is None:
                    continue
                npred += 1; tp += abs(p + lag - GT_EV[s]) <= TOL
            prec = tp / npred if npred else 0; rec = tp / len(SEQS)
            f1s.append(2 * prec * rec / (prec + rec) if tp else 0.0)
        return float(np.mean(f1s))
    best = None
    for rho in np.round(np.arange(0.1, 0.95, 0.05), 2):
        diffs = [GT_EV[s] - p for s in SEQS for (ap, _, fr) in traces_by_cond["G0"][s]
                 if (p := onset(ap, fr, rho)) is not None]
        if not diffs:
            continue
        lag = int(round(np.median(diffs))); f1 = f1_of("G0", rho, lag)
        if best is None or f1 > best[0]:
            best = (f1, rho, lag)
    _, rho, lag = best
    return {c: f1_of(c, rho, lag) for c in traces_by_cond}, rho, lag


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    # One-Euro params, tuned on WiLoR G0 only: minimise 2-view jitter subject to <= 5% increase in
    # 2-view absolute MPJPE over the unfiltered pipeline (pure MPJPE minimisation picks "no filter").
    base = [run_2view("wilor", "G0", s)[0] for s in SEQS]
    base_e = np.nanmean([b["mpjpe_abs_mm"] for b in base])
    grid = list(itertools.product([1.0, 2.0, 4.0, 8.0, 16.0], [0.0, 0.005, 0.02, 0.05, 0.2]))
    tune = []
    for mc, beta in grid:
        r = [run_2view("wilor", "G0", s, (mc, beta))[0] for s in SEQS]
        e, j = np.nanmean([x["mpjpe_abs_mm"] for x in r]), np.nanmean([x["jitter_mm_f2"] for x in r])
        tune.append((j, e, mc, beta))
    ok = [t for t in tune if t[1] <= 1.05 * base_e]
    j, e, *OE = min(ok) if ok else min(tune, key=lambda t: t[1]); OE = tuple(OE)
    print(f"One-Euro tuned on WiLoR G0 (mincutoff, beta) = {OE}: MPJPE {e:.2f} (unfiltered {base_e:.2f}) mm, jitter {j:.2f}")
    import json
    json.dump({"one_euro": {"mincutoff": OE[0], "beta": OE[1], "dcutoff": 1.0, "g0_mpjpe_abs_mm": e,
                            "g0_mpjpe_abs_unfiltered_mm": base_e, "g0_jitter_mm_f2": j}},
              open(OUT / "p3_params.json", "w"), indent=1)

    rows, gate_frames, ev_traces = [], [], {}
    for model in MODELS:
        for cond in CONDS:
            for seq in SEQS:
                mono, mtr = run_mono(model, cond, seq)
                tv, ttr, re, vis = run_2view(model, cond, seq)
                oe, otr, _, _ = run_2view(model, cond, seq, OE)
                hy, htr = run_hybrid(model, cond, seq)
                for pipe, r in (("mono", mono), ("2view", tv), ("2view_oe", oe), ("hybrid", hy)):
                    rows.append({"model": model, "pipeline": pipe, "cond": cond, "seq": seq, **r})
                ev_traces.setdefault((model, "mono"), {}).setdefault(cond, {})[seq] = list(mtr.values())
                ev_traces.setdefault((model, "2view"), {}).setdefault(cond, {})[seq] = [ttr]
                ev_traces.setdefault((model, "2view_oe"), {}).setdefault(cond, {})[seq] = [otr]
                ev_traces.setdefault((model, "hybrid"), {}).setdefault(cond, {})[seq] = [htr]
                gate_frames.append({"model": model, "cond": cond, "seq": seq, "re": re, "vis": vis})
    pipes = pd.DataFrame(rows)
    ub = pd.read_csv(OUT / "upper_bound.csv")
    ub = ub[ub.seq.isin(SEQS)]
    for _, r in ub.iterrows():
        rows.append({"model": "wilor", "pipeline": "8cam", "cond": r.cond, "seq": r.seq,
                     "mpjpe_abs_mm": r.mpjpe_abs_8cam_mm, "mpjpe_rel_mm": np.nan, "jitter_mm_f2": np.nan})
    pipes = pd.DataFrame(rows); pipes.to_csv(OUT / "p3_pipelines.csv", index=False, float_format="%.4g")

    ev_rows = []
    for (model, pipe), tbc in ev_traces.items():
        f1, rho, lag = event_scores(tbc)
        for c, v in f1.items():
            ev_rows.append({"model": model, "pipeline": pipe, "cond": c, "f1": v, "rho": rho, "lag_frames": lag})
    ev = pd.DataFrame(ev_rows); ev.to_csv(OUT / "p3_events.csv", index=False, float_format="%.3g")

    # ---- gate
    g0_re = np.concatenate([list(g["re"].values()) for g in gate_frames if g["model"] == "wilor" and g["cond"] == "G0"])
    tau = float(np.percentile(g0_re, 95))
    gate = []
    err2 = pipes[pipes.pipeline == "2view"].set_index(["model", "cond", "seq"]).mpjpe_abs_mm
    for X in [50, 60, 70, 80, 90, 95]:
        for g in gate_frames:
            n = max(len(g["vis"]), 1)
            valid = sum(1 for f in g["vis"] if f in g["re"] and g["re"][f] < tau) / n
            gate.append({"tau_px": tau, "X": X, "model": g["model"], "cond": g["cond"], "seq": g["seq"],
                         "valid_frac": valid, "pass": int(valid >= X / 100),
                         "mpjpe_abs_mm": err2.get((g["model"], g["cond"], g["seq"]), np.nan)})
    gate = pd.DataFrame(gate); gate.to_csv(OUT / "p3_gate.csv", index=False, float_format="%.4g")
    print(f"gate tau = {tau:.1f} px")
    prm = json.load(open(OUT / "p3_params.json")); prm["gate_tau_px"] = tau
    json.dump(prm, open(OUT / "p3_params.json", "w"), indent=1)
    print(gate[gate.model == "wilor"].pivot_table(index="cond", columns="X", values="pass", aggfunc="mean").round(2).reindex(CONDS))
    pf = gate[(gate.model == "wilor") & (gate.X == 80)]
    print("WiLoR X=80: MPJPE pass vs fail:", pf.groupby("pass").mpjpe_abs_mm.mean().round(1).to_dict())

    # ---- recovery table (WiLoR): % of the G0->Gx degradation of the mono baseline recovered
    def seqmean(pipe, cond, met, model="wilor"):
        return pipes[(pipes.model == model) & (pipes.pipeline == pipe) & (pipes.cond == cond)].set_index("seq")[met]

    def gap_reduction(base_x, base_0, int_x, int_0):
        """1 - (int_x - int_0) / (base_x - base_0): does the intervention make the system less
        glove-sensitive? 100% = no glove degradation left, 0% = same degradation as baseline."""
        df = pd.concat([base_x, base_0, int_x, int_0], axis=1).dropna().values
        if len(df) < 3:
            return np.nan, np.nan, np.nan
        def r(d):
            den = d[:, 0].mean() - d[:, 1].mean()
            return 100 * (1 - (d[:, 2].mean() - d[:, 3].mean()) / den) if abs(den) > 1e-9 else np.nan
        bs = [r(df[rng.integers(0, len(df), len(df))]) for _ in range(CFG["analysis"]["bootstrap"]["n"])]
        return r(df), *np.nanpercentile(bs, [2.5, 97.5])

    def recovery(base_x, base_0, int_x):
        """bootstrap over sequences of (mean(base_x) - mean(int_x)) / (mean(base_x) - mean(base_0))"""
        df = pd.concat([base_x, base_0, int_x], axis=1).dropna().values
        if len(df) < 3:
            return np.nan, np.nan, np.nan
        def r(d):
            den = d[:, 0].mean() - d[:, 1].mean()
            return 100 * (d[:, 0].mean() - d[:, 2].mean()) / den if abs(den) > 1e-9 else np.nan
        bs = [r(df[rng.integers(0, len(df), len(df))]) for _ in range(CFG["analysis"]["bootstrap"]["n"])]
        return r(df), *np.nanpercentile(bs, [2.5, 97.5])

    rec = []
    for cond in ["G2", "G3a", "G3b_k4"]:
        for met in ["mpjpe_abs_mm", "mpjpe_rel_mm", "jitter_mm_f2"]:
            for pipe in ["2view", "2view_oe", "hybrid", "8cam"]:
                if pipe == "8cam" and (met != "mpjpe_abs_mm" or cond not in CFG["analysis"]["upper_bound"]["conditions"]):
                    continue
                bx, b0, ix, i0 = seqmean("mono", cond, met), seqmean("mono", "G0", met), seqmean(pipe, cond, met), seqmean(pipe, "G0", met)
                v = recovery(bx, b0, ix); g = gap_reduction(bx, b0, ix, i0)
                rec.append({"cond": cond, "metric": met, "intervention": f"mono -> {pipe}", "recovery_pct": v[0], "ci_lo": v[1], "ci_hi": v[2],
                            "gap_reduction_pct": g[0], "gr_lo": g[1], "gr_hi": g[2]})
            if cond != "G1":   # glove spec: same pipeline (2view), thin bright nitrile G1 instead of Gx (colour AND thickness differ; G4 isolates colour)
                v = recovery(seqmean("2view", cond, met), seqmean("2view", "G0", met), seqmean("2view", "G1", met))
                rec.append({"cond": cond, "metric": met, "intervention": "glove spec: thin nitrile G1 instead of Gx (2view)",
                            "recovery_pct": v[0], "ci_lo": v[1], "ci_hi": v[2]})
        # events (higher is better; no CI: one binary outcome per sequence, F1 is a set statistic)
        e = ev[ev.model == "wilor"].set_index(["pipeline", "cond"]).f1
        base_x, base_0 = e[("mono", cond)], e[("mono", "G0")]
        for pipe in ["2view", "2view_oe", "hybrid"]:
            den = base_0 - base_x
            rec.append({"cond": cond, "metric": "event_f1", "intervention": f"mono -> {pipe}",
                        "recovery_pct": 100 * (e[(pipe, cond)] - base_x) / den if abs(den) > 1e-9 else np.nan,
                        "ci_lo": np.nan, "ci_hi": np.nan})
    # model swap for detection (per-view detection rate, from P2 results)
    res = pd.read_csv(REP / "results.csv")
    det = res[(res.scope == "per_view_detected") & (res.metric == "det_rate")].set_index(["model", "cond"])["mean"]
    for cond in ["G2", "G3a", "G3b_k4"]:
        den = det[("mediapipe", "G0")] - det[("mediapipe", cond)]
        rec.append({"cond": cond, "metric": "det_rate", "intervention": "model: MediaPipe -> WiLoR",
                    "recovery_pct": 100 * (det[("wilor", cond)] - det[("mediapipe", cond)]) / den, "ci_lo": np.nan, "ci_hi": np.nan})
    rec = pd.DataFrame(rec); rec.to_csv(REP / "recovery.csv", index=False, float_format="%.3g")

    summ = pipes.groupby(["model", "pipeline", "cond"])[["mpjpe_abs_mm", "mpjpe_rel_mm", "jitter_mm_f2", "root_z_bias_mm"]].mean().round(2)
    summ.to_csv(OUT / "p3_pipeline_summary.csv")
    print(summ.loc["wilor"].reindex(CONDS, level=1))
    print(ev[ev.model == "wilor"].pivot_table(index="cond", columns="pipeline", values="f1").round(2).reindex(CONDS))
    print(rec.round(1).to_string())
