"""Standardized per-image hand keypoint record (MANO/OpenPose 21-joint order, wrist=0;
same order as DexYCB joint_3d) and a shared driver for smoke-test and batch modes."""
import csv, json, pathlib, sys, time
from collections import defaultdict

N_JOINTS = 21
ROOT = pathlib.Path(__file__).resolve().parents[1]


def record(model, image, hands, runtime_ms=None):
    """hands: list of dict(handedness, kp2d[21][2] px, kp3d[21][3] or None, conf)."""
    return {"model": model, "image": image, "n_joints": N_JOINTS,
            "runtime_ms": runtime_ms, "hands": hands}


def dump(rec, path):
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


def pred_path(model, cond, seq, cam):
    return ROOT / "data/cache/pred" / model / cond / seq / f"{cam}.jsonl"


def groups(frames_csv, pair_only=False):
    """frames.csv -> {(cond, seq, cam): [(frame, path), ...]}; pair_only keeps the 2-view rig."""
    import yaml
    pair = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))["dexycb"]["pair"]
    g = defaultdict(list)
    for r in csv.DictReader(open(frames_csv)):
        if pair_only and r["cam"] not in pair:
            continue
        g[(r["condition"], r["seq"], r["cam"])].append((int(r["frame"]), r["path"]))
    return g


def run_group(model, key, items, predict, imread):
    """Predict every frame of one (cond, seq, cam) group -> JSONL. Skips finished groups."""
    out = pred_path(model, *key)
    if out.exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f, p in sorted(items):
        img = imread(str(ROOT / p))
        t = time.time()
        hands = predict(img, key, f)
        rec = record(model, p, hands, (time.time() - t) * 1e3)
        rec["frame"] = f
        lines.append(json.dumps(rec))
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(out)  # atomic: a killed run never leaves a half-written cache file
    return len(lines)


def smoke(model, img_dir, out_dir, predict, imread):
    img_dir, out_dir = pathlib.Path(img_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(img_dir.glob("*")):
        img = imread(str(p))
        if img is None:
            continue
        t = time.time()
        hands = predict(img, None, p.stem)
        ms = (time.time() - t) * 1e3
        dump(record(model, p.name, hands, ms), out_dir / f"{p.stem}.{model}.json")
        print(p.name, len(hands), "hands", f"{ms:.0f} ms")
