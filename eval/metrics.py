"""P2 metrics vs DexYCB ground truth -> eval/out/{per_frame,per_seq_mono,per_seq_2view,events,upper_bound}.csv

Per view (pair cams), per frame with the hand visible in GT (mask >= 500 px, valid joints):
  detected (same matching rule for every model), 2D EPE + PCK@0.2 (threshold = 0.2 x GT keypoint
  box size); WiLoR/HaMeR also root-relative MPJPE, PA-MPJPE, absolute 3D error after re-solving
  translation with the true intrinsics, wrist depth vs the dataset depth map, aperture error.
Per sequence, 2-view rig: DLT triangulation of each model's 2D keypoints (MediaPipe gets metric
  3D only this way): absolute and root-relative MPJPE, cross-view reprojection error, jitter,
  thumb-index aperture trace -> grasp-onset events.
Usage: metrics.py
"""
import csv, sys
import numpy as np
from common import (ROOT, CFG, PAIR, FPS, gt, depth, load_pred, match, box, procrustes,
                    solve_translation, triangulate, proj_matrix, reproj_err, intrinsics,
                    world_from_cam, sequences, conditions)

MODELS = ["mediapipe", "wilor", "hamer"]
OUT = ROOT / "eval/out"; OUT.mkdir(parents=True, exist_ok=True)
TIP_T, TIP_I = 4, 8


def nan():
    return float("nan")


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))  # union: rows differ by model
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def per_view(seqs, conds):
    rows = []
    for model in MODELS:
        for cond in conds:
            for seq in seqs:
                for cam in PAIR:
                    pred = load_pred(model, cond, seq, cam)
                    if pred is None:
                        continue   # e.g. MediaPipe on G4* (not run)
                    K = intrinsics(cam)
                    for f in sorted(pred):
                        j3, j2, vis = gt(seq, cam, f)
                        if not vis:
                            continue
                        h = match(pred[f], j2)
                        r = {"model": model, "cond": cond, "seq": seq, "cam": cam, "frame": f,
                             "detected": int(h is not None)}
                        if h is not None:
                            k2 = np.asarray(h["kp2d"]); e2 = np.linalg.norm(k2 - j2, axis=1)
                            bb = box(j2); thr = 0.2 * max(bb[2] - bb[0], bb[3] - bb[1])
                            r.update(epe2d_px=e2.mean(), pck02=(e2 < thr).mean())
                            if "kp3d_rel" in h:
                                rel = np.asarray(h["kp3d_rel"]); grel = j3 - j3[0]
                                r["mpjpe_rel_mm"] = 1000 * np.linalg.norm(rel - grel, axis=1).mean()
                                r["pa_mpjpe_mm"] = 1000 * np.linalg.norm(procrustes(rel, grel) - grel, axis=1).mean()
                                t = solve_translation(rel, k2, K); a = rel + t
                                r["mpjpe_abs_mm"] = 1000 * np.linalg.norm(a - j3, axis=1).mean()
                                r["root_err_mm"] = 1000 * np.linalg.norm(a[0] - j3[0])
                                d = depth(seq, cam, f); u, v = np.clip(k2[0].round().astype(int), [2, 2], [637, 477])
                                patch = d[v - 2:v + 3, u - 2:u + 3]; patch = patch[patch > 0]
                                if patch.size:
                                    z = np.median(patch) / 1000.0
                                    r["wrist_depth_err_mm"] = 1000 * abs(a[0, 2] - z)
                                    r["gt_wrist_depth_err_mm"] = 1000 * abs(j3[0, 2] - z)  # sensor/surface floor
                                ap, gap = (np.linalg.norm(rel[TIP_T] - rel[TIP_I]), np.linalg.norm(grel[TIP_T] - grel[TIP_I]))
                                r["aperture_err_mm"] = 1000 * abs(ap - gap)
                        rows.append(r)
    return rows


def tri_frame(hands_by_cam, seq, f):
    """2-view (or n-view) DLT from matched 2D keypoints; None unless >= 2 views detected."""
    k2s, Ps, cams = [], [], []
    for cam, hands in hands_by_cam.items():
        _, j2, vis = gt(seq, cam, f)
        h = match(hands.get(f, []), j2) if vis or True else None
        # matching needs a GT box; use it even if the view's mask is small (occluded) -- the
        # model may still see the hand there. A view without valid GT joints can't be matched.
        if h is None or (j2 < 0).any():
            continue
        k2s.append(np.asarray(h["kp2d"])); Ps.append(proj_matrix(seq, cam)); cams.append(cam)
    if len(k2s) < 2:
        return None
    return triangulate(k2s, Ps), k2s, Ps, cams


def gt_world(seq, cam, f):
    j3, _, _ = gt(seq, cam, f)
    T = world_from_cam(seq, cam)
    return (T[:3, :3] @ j3.T).T + T[:3, 3]


def two_view(seqs, conds):
    rows, traces = [], {}
    for model in MODELS:
        for cond in conds:
            for seq in seqs:
                preds = {cam: load_pred(model, cond, seq, cam) for cam in PAIR}
                if any(p is None for p in preds.values()):
                    continue
                frames = sorted(set.intersection(*[set(p) for p in preds.values()]))
                X, G = {}, {}
                re = []
                for f in frames:
                    G[f] = gt_world(seq, PAIR[0], f)
                    t = tri_frame(preds, seq, f)
                    if t is None:
                        continue
                    X[f], k2s, Ps, _ = t
                    re.append(np.mean([reproj_err(X[f], k, P).mean() for k, P in zip(k2s, Ps)]))
                n_eval = sum(any(gt(seq, c, f)[2] for c in PAIR) for f in frames)
                ok = sorted(X)
                abs_e = [1000 * np.linalg.norm(X[f] - G[f], axis=1).mean() for f in ok]
                rel_e = [1000 * np.linalg.norm((X[f] - X[f][0]) - (G[f] - G[f][0]), axis=1).mean() for f in ok]
                # jitter: mean joint acceleration over runs of consecutive triangulated frames
                acc, gacc = [], []
                for f in ok:
                    if f - 1 in X and f + 1 in X:
                        acc.append(1000 * np.linalg.norm(X[f + 1] - 2 * X[f] + X[f - 1], axis=1).mean())
                        gacc.append(1000 * np.linalg.norm(G[f + 1] - 2 * G[f] + G[f - 1], axis=1).mean())
                ap = {f: 1000 * np.linalg.norm(X[f][TIP_T] - X[f][TIP_I]) for f in ok}
                gap = {f: 1000 * np.linalg.norm(G[f][TIP_T] - G[f][TIP_I]) for f in frames}
                traces[(model, cond, seq)] = (ap, gap, frames)
                rows.append({"model": model, "cond": cond, "seq": seq,
                             "tri_rate": len(ok) / max(n_eval, 1),
                             "mpjpe_abs_mm": np.mean(abs_e) if ok else nan(),
                             "mpjpe_rel_mm": np.mean(rel_e) if ok else nan(),
                             "reproj_px": np.mean(re) if re else nan(),
                             "jitter_mm_f2": np.mean(acc) if acc else nan(),
                             "gt_jitter_mm_f2": np.mean(gacc) if gacc else nan(),
                             "aperture_err_mm": np.mean([abs(ap[f] - gap[f]) for f in ok]) if ok else nan()})
    return rows, traces


# ---------------------------------------------------------------- grasp-onset events
def onset(ap, frames, rho, max_gap=5):
    """Aperture-based grasp onset: interpolate gaps <= max_gap frames, median-filter (5), then the
    first frame after the aperture peak where a(t) <= a_end + rho * (a_peak - a_end); a_end = median
    of the last 10 frames (hand holding the object). Per-sequence normalisation, because the
    final aperture equals the object's width. Offline (uses future frames), like log matching."""
    f0, f1 = frames[0], frames[-1]
    t = np.arange(f0, f1 + 1); have = np.array([f in ap for f in t])
    if have.sum() < 10:
        return None
    a = np.interp(t, t[have], [ap[f] for f in t[have]])
    # mask long gaps: no events inside them
    gap_len = np.zeros(len(t), int); run = 0
    for i, h in enumerate(have):
        run = 0 if h else run + 1; gap_len[i] = run
    from scipy.ndimage import median_filter
    a = median_filter(a, 5, mode="nearest")
    ip = int(np.argmax(a)); a_end = np.median(a[-10:]); thr = a_end + rho * (a[ip] - a_end)
    for i in range(ip, len(t)):
        if a[i] <= thr and gap_len[i] <= max_gap:
            return int(t[i])
    return None


def events(traces, seqs, conds):
    gt_ev = {r["seq"]: int(r["frame"]) for r in csv.DictReader(open(ROOT / "data/events.csv"))}
    tol = round(CFG["events"]["tolerance_s"] * FPS)
    rhos = np.round(np.arange(0.1, 0.95, 0.05), 2)

    def score(model, cond, rho, lag, use_gt_trace=False):
        tp, npred, errs = 0, 0, []
        for s in seqs:
            ap, gap, frames = traces[(model, cond, s)]
            p = onset(gap if use_gt_trace else ap, frames, rho)
            if p is None:
                continue
            npred += 1; e = p + lag - gt_ev[s]; errs.append(e)
            tp += abs(e) <= tol
        prec = tp / npred if npred else 0.0; rec = tp / len(seqs)
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        return f1, prec, rec, errs

    rows, tuned = [], {}
    for model in MODELS + ["gt_oracle"]:
        m = "wilor" if model == "gt_oracle" else model
        use_gt = model == "gt_oracle"
        best = None
        for rho in rhos:   # tune rho and the constant lag on G0 only
            d = [onset(traces[(m, "G0", s)][1 if use_gt else 0], traces[(m, "G0", s)][2], rho) for s in seqs]
            diffs = [gt_ev[s] - p for s, p in zip(seqs, d) if p is not None]
            if not diffs:
                continue
            lag = int(round(np.median(diffs)))
            f1 = score(m, "G0", rho, lag, use_gt)[0]
            if best is None or f1 > best[0]:
                best = (f1, rho, lag)
        _, rho, lag = best; tuned[model] = (rho, lag)
        for cond in (["G0"] if use_gt else conds):
            if any((m, cond, s) not in traces for s in seqs):
                continue   # model not run on this condition (G4*: WiLoR/HaMeR only)
            f1, prec, rec, errs = score(m, cond, rho, lag, use_gt)
            rows.append({"model": model, "cond": cond, "rho": rho, "lag_frames": lag,
                         "lag_ms": round(1000 * lag / FPS), "f1": f1, "precision": prec, "recall": rec,
                         "n_pred": len(errs), "median_abs_err_ms": 1000 * np.median(np.abs(errs)) / FPS if errs else nan()})
    return rows, tuned


def upper_bound(seqs):
    ub = CFG["analysis"]["upper_bound"]
    serials = sorted(world_from_cam(seqs[0], PAIR[0]) is not None and
                     ["836212060125", "839512060362", "840412060917", "841412060263",
                      "932122060857", "932122060861", "932122061900", "932122062010"])
    rows = []
    for cond in ub["conditions"]:
        for seq in seqs:
            preds = {c: load_pred(ub["model"], cond, seq, c) for c in serials}
            preds = {c: p for c, p in preds.items() if p is not None}
            frames = sorted(set.intersection(*[set(p) for p in preds.values()]))
            e8, e2, n8, n2 = [], [], 0, 0
            for f in frames:
                G = gt_world(seq, PAIR[0], f)
                for sub, acc in ((preds, e8), ({c: preds[c] for c in PAIR}, e2)):
                    t = tri_frame(sub, seq, f)
                    if t is None:
                        continue
                    X, k2s, Ps, cams = t
                    if len(cams) > 2:   # one robustness pass: drop the worst view if it disagrees
                        err = [reproj_err(X, k, P).mean() for k, P in zip(k2s, Ps)]
                        w = int(np.argmax(err))
                        if err[w] > 20:
                            keep = [i for i in range(len(cams)) if i != w]
                            X = triangulate([k2s[i] for i in keep], [Ps[i] for i in keep])
                    acc.append(1000 * np.linalg.norm(X - G, axis=1).mean())
            rows.append({"cond": cond, "seq": seq, "n_frames": len(frames),
                         "rate_8cam": len(e8) / len(frames), "mpjpe_abs_8cam_mm": np.mean(e8) if e8 else nan(),
                         "rate_2cam": len(e2) / len(frames), "mpjpe_abs_2cam_mm": np.mean(e2) if e2 else nan()})
    return rows


if __name__ == "__main__":
    seqs_all = sequences(include_excluded=True)
    conds = conditions()
    write("per_frame.csv", per_view(seqs_all, conds))
    rows, traces = two_view(seqs_all, conds)
    write("per_seq_2view.csv", rows)
    import pickle
    pickle.dump(traces, open(OUT / "traces.pkl", "wb"))
    ev, tuned = events(traces, sequences(), conds)   # tuned + scored on main set only
    write("events.csv", ev)
    print("event tuning (rho, lag frames):", tuned)
    write("upper_bound.csv", upper_bound(seqs_all))
