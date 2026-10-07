"""Milestone figure: same frames under every glove condition.
Usage: glove_grid.py OUT_JPG SEQ_DIR:FRAME [SEQ_DIR:FRAME ...]  (rows = seq x pair cam)"""
import sys, pathlib, cv2, yaml, numpy as np
from synth_gloves import apply_glove, conditions, refine_mask

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
out = sys.argv[1]
rows = []
for arg in sys.argv[2:]:
    seq, frame = pathlib.Path(arg.rsplit(":", 1)[0]), int(arg.rsplit(":", 1)[1])
    for cam in cfg["dexycb"]["pair"]:
        img = cv2.imread(str(seq / cam / f"color_{frame:06d}.jpg"))
        lab = np.load(seq / cam / f"labels_{frame:06d}.npz")
        mask = refine_mask(img, lab["seg"], lab["joint_2d"][0])
        ys, xs = np.nonzero(mask)
        c = (int(xs.mean()), int(ys.mean())); r = 110   # crop around hand for visibility
        x0, y0 = np.clip(c[0] - r, 0, 640 - 2 * r), np.clip(c[1] - r, 0, 480 - 2 * r)
        tiles = []
        for name, spec, k in conditions(cfg["gloves"]):
            g = apply_glove(img, mask, spec, seed=cfg["seed"], grow_px=k)[y0:y0 + 2 * r, x0:x0 + 2 * r]
            g = cv2.resize(g, (300, 300), interpolation=cv2.INTER_NEAREST)
            cv2.putText(g, name, (6, 24), 0, 0.7, (0, 255, 255), 2); tiles.append(g)
        rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows))
