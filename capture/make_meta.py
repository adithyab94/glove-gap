"""Select DexYCB sequences and write data/meta.csv (one row per sequence x camera x condition).

Mapping to the original capture plan: task = grasped YCB object, rep = index among this
subject's sequences with that object, condition = synthetic glove. Validates completeness
(every selected sequence has all frames, labels and depth for both pair cameras).
Usage: make_meta.py
"""
import csv, pathlib, sys, yaml
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "capture"))
from synth_gloves import conditions

# YCB ids used by DexYCB (dex-ycb-toolkit dex_ycb_toolkit/dex_ycb.py _YCB_CLASSES)
YCB = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can",
       5: "006_mustard_bottle", 6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box",
       9: "010_potted_meat_can", 10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
       13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 17: "037_scissors",
       18: "040_large_marker", 19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick"}

cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
d = cfg["dexycb"]
subj = ROOT / d["root"] / d["subject"]
rows, problems, seen = [], [], {}
allseq = sorted(p for p in subj.iterdir() if p.is_dir())
assert len(allseq) == 100, "toolkit split indices assume 100 sequences per subject"
# s0_heldout: DexYCB S0 holds out sequence i (sorted) iff i % 5 == 4; everything else is S0-train.
sel = [(s, yaml.safe_load(open(s / "meta.yml"))) for i, s in enumerate(allseq) if i % 5 == 4]
conds = [c[0] for c in conditions(cfg["gloves"])]
for seq, meta in sel:
    obj = YCB[meta["ycb_ids"][meta["ycb_grasp_ind"]]]
    rep = seen[obj] = seen.get(obj, -1) + 1
    n = meta["num_frames"]
    for cam in d["pair"]:
        cdir = seq / cam
        for kind, ext in (("color", "jpg"), ("labels", "npz"), ("aligned_depth_to_color", "png")):
            missing = [i for i in range(n) if not (cdir / f"{kind}_{i:06d}.{ext}").exists()]
            if missing:
                problems.append(f"{seq.name}/{cam}: {len(missing)} missing {kind}")
        hand_vis = np.mean([(np.load(cdir / f"labels_{i:06d}.npz")["seg"] == 255).sum() > 500
                            for i in range(0, n, 6)]) if not problems else float("nan")
        for c in conds:
            rows.append({"seq": seq.name, "cam": cam, "condition": c, "task": obj, "rep": rep,
                         "side": meta["mano_sides"][0],
                         "n_frames": n, "hand_visible_frac": round(float(hand_vis), 3),
                         "src_dir": str(cdir.relative_to(ROOT)),
                         "img_dir": f"data/cache/gloved/{c}/{seq.name}/{cam}"})
out = ROOT / "data/meta.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"selected {len(sel)} S0-held-out sequences "
      f"({sum(m['mano_sides'] == ['right'] for _, m in sel)} right); {len(rows)} rows -> {out}")
print("objects:", {k: v + 1 for k, v in seen.items()})
print("COMPLETE" if not problems else "INCOMPLETE:\n  " + "\n  ".join(problems))
sys.exit(1 if problems else 0)
