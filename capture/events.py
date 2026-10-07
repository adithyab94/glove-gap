"""Weak grasp labels from object motion (stand-in for warehouse pick logs).

Event = first frame where the grasped YCB object's translation has moved more than
move_thresh_m from its pose at frame 0. DexYCB sequences are grasp-and-lift (no release),
so there is one grasp event per sequence. The constant onset-vs-contact lag is estimated
on G0 in eval/ and applied to every condition.
Usage: events.py  -> data/events.csv
"""
import csv, pathlib, yaml
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
thr, fps = cfg["events"]["move_thresh_m"], cfg["dexycb"]["fps_src"]
subj = ROOT / cfg["dexycb"]["root"] / cfg["dexycb"]["subject"]
seqs = sorted({r["seq"] for r in csv.DictReader(open(ROOT / "data/meta.csv"))})
rows = []
for s in seqs:
    meta = yaml.safe_load(open(subj / s / "meta.yml"))
    t = np.load(subj / s / "pose.npz")["pose_y"][:, meta["ycb_grasp_ind"], 4:]  # (T, 3) xyz, m
    disp = np.linalg.norm(t - t[0], axis=1)
    hit = np.nonzero(disp > thr)[0]
    f = int(hit[0]) if len(hit) else -1
    rows.append({"seq": s, "event": "grasp_onset", "frame": f, "t_s": round(f / fps, 4) if f >= 0 else "",
                 "max_disp_m": round(float(disp.max()), 4)})
with open(ROOT / "data/events.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
miss = [r["seq"] for r in rows if r["frame"] < 0]
print(f"{len(rows)} sequences, {len(rows) - len(miss)} with onset; no onset: {miss}")
