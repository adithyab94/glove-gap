"""Quantify skin leakage outside DexYCB hand masks (MANO-fit masks miss real skin at edges).
Leak = fraction of pixels in a 1..R px ring outside the mask, near finger/palm joints (not the
wrist/forearm), that a YCrCb skin rule classifies as skin. Usage: mask_leak.py SEQ_DIR [SEQ_DIR...]"""
import sys, pathlib, cv2, yaml, numpy as np
from synth_gloves import grow_mask, refine_mask, skin_rule

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
R, NEAR, WRIST_EXCL = 4, 30, 45   # ring width, max dist to finger joint, min dist to wrist (px)
res = {}
for seq in map(pathlib.Path, sys.argv[1:]):
    for cam in cfg["dexycb"]["pair"]:
        for lf in sorted((seq / cam).glob("labels_*.npz"))[::6]:
            lab = np.load(lf)
            j2 = lab["joint_2d"].reshape(-1, 2)
            if not (lab["seg"] == 255).any() or (j2 < 0).any():
                continue
            img = cv2.imread(str(lf).replace("labels_", "color_").replace(".npz", ".jpg"))
            skin = skin_rule(img)
            near = np.zeros(lab["seg"].shape, np.uint8)
            for x, y in j2[1:]:
                cv2.circle(near, (int(x), int(y)), NEAR, 1, -1)
            cv2.circle(near, tuple(j2[0].astype(int)), WRIST_EXCL, 0, -1)
            for kind, mask in (("raw", lab["seg"] == 255), ("refined", refine_mask(img, lab["seg"], j2))):
                # only background-labelled ring px count (objects can be skin-coloured too)
                r = grow_mask(mask, R) & ~mask & near.astype(bool) & (lab["seg"] == 0)
                if r.sum() > 20:
                    res.setdefault((cam, kind), []).append(float((skin & r).sum() / r.sum()))
for (cam, kind), v in sorted(res.items()):
    v = np.array(v)
    print(f"{cam} {kind:8s}: n={len(v)} frames  skin-in-ring median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}")
