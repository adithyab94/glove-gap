"""Render synthetic-glove frames for P2 and write data/cache/frames.csv.

Pair cams: every condition, every frame. Other 6 cams (8-cam upper bound): the conditions and
frame stride in configs analysis.upper_bound. All conditions, G0 included, are re-encoded as
JPEG q95 so compression is identical. Per frame we log mask size and the residual skin share
in a 4 px ring around the refined mask (near the fingers, background pixels only).
Usage: render.py  (idempotent: existing JPEGs are kept)
"""
import csv, pathlib, sys, yaml
from multiprocessing import Pool
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "capture"))
from synth_gloves import apply_glove, conditions, grow_mask, refine_mask, skin_rule

cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
CONDS = {n: (s, k) for n, s, k in conditions(cfg["gloves"])}
SERIALS = ["836212060125", "839512060362", "840412060917", "841412060263",
           "932122060857", "932122060861", "932122061900", "932122062010"]


def job(args):
    seq, cam, frames, conds = args
    src = ROOT / cfg["dexycb"]["root"] / cfg["dexycb"]["subject"] / seq / cam
    rows = []
    for f in frames:
        img = cv2.imread(str(src / f"color_{f:06d}.jpg"))
        lab = np.load(src / f"labels_{f:06d}.npz")
        seg, j2 = lab["seg"], lab["joint_2d"][0]
        mask_px = int((seg == 255).sum())
        valid_j = bool((j2 >= 0).all())
        mask = refine_mask(img, seg, j2 if valid_j and mask_px > 200 else None)
        skin = ""
        if valid_j and mask_px > 500:
            hull = np.zeros(seg.shape, np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(j2.astype(np.int32)), 1)
            ring = grow_mask(mask, 4) & ~mask & (seg == 0) & grow_mask(hull.astype(bool), 10)
            if ring.sum() > 20:
                skin = round(float((skin_rule(img) & ring).sum() / ring.sum()), 4)
        for c in conds:
            spec, k = CONDS[c]
            out = ROOT / "data/cache/gloved" / c / seq / cam / f"color_{f:06d}.jpg"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                # constant seed: texture must not change between frames (would add fake flicker/jitter)
                g = apply_glove(img, mask, spec, seed=cfg["seed"], grow_px=k)
                cv2.imwrite(str(out), g, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"condition": c, "seq": seq, "cam": cam, "frame": f, "mask_px": mask_px,
                         "skin_ring": skin, "path": str(out.relative_to(ROOT))})
    return rows


if __name__ == "__main__":
    meta = list(csv.DictReader(open(ROOT / "data/meta.csv")))
    seqs = sorted({(r["seq"], int(r["n_frames"])) for r in meta})
    ub = cfg["analysis"]["upper_bound"]
    jobs = []
    for seq, n in seqs:
        for cam in SERIALS:
            if cam in cfg["dexycb"]["pair"]:
                jobs.append((seq, cam, list(range(n)), list(CONDS)))  # G4* included (added in P4)
            else:
                jobs.append((seq, cam, list(range(0, n, ub["stride"])), ub["conditions"]))
    with Pool(20) as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    rows.sort(key=lambda r: (r["condition"], r["seq"], r["cam"], r["frame"]))
    with open(ROOT / "data/cache/frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} frames rendered/indexed")
