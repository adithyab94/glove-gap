"""Qualitative figures for the README (from cached predictions):
  report/fig_pose_examples.jpg  one DexYCB frame per glove condition, GT (white) vs WiLoR (orange)
                                vs MediaPipe (blue) 2D skeletons, cropped around the hand
  report/fig_real_detection.jpg Roboflow worn-glove test images: GT box (white) + WiLoR / MediaPipe
                                skeletons; first 4 hits and first 4 misses of WiLoR by image id
Selection is by rule (fixed sequence/frame; first-N by id), not by eye. Usage: examples.py"""
import json, pathlib, sys
import cv2, numpy as np, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from common import gt, load_pred, match, PAIR  # noqa: E402

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
BGR = {"gt": (255, 255, 255), "wilor": (52, 104, 235), "mediapipe": (214, 120, 42)}   # #eb6834, #2a78d6


def skel(img, k, col, t=2):
    k = np.asarray(k).astype(int)
    for a, b in BONES:
        cv2.line(img, tuple(k[a]), tuple(k[b]), col, t, cv2.LINE_AA)
    for p in k:
        cv2.circle(img, tuple(p), t + 1, col, -1, cv2.LINE_AA)


def label(img, text, y=22, scale=0.6):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def pose_examples(seq="20200709_142802", cam=PAIR[1], frame=40, out="fig_pose_examples.jpg"):
    conds = [("G0", "bare"), ("G1", "thin nitrile"), ("G2", "grey knit"), ("G3a", "dark insulated"),
             ("G3b_k4", "dark ins. +4px"), ("G4", "bright insulated")]
    _, j2, _ = gt(seq, cam, frame)
    c = j2.mean(0); r = int(max(np.ptp(j2, 0)) * 0.9) + 20
    x0, y0 = int(np.clip(c[0] - r, 0, 640 - 2 * r)), int(np.clip(c[1] - r, 0, 480 - 2 * r))
    tiles = []
    for cond, name in conds:
        img = cv2.imread(str(ROOT / f"data/cache/gloved/{cond}/{seq}/{cam}/color_{frame:06d}.jpg"))
        skel(img, j2, BGR["gt"], 1)
        status = []
        for model in ["mediapipe", "wilor"]:
            p = load_pred(model, cond, seq, cam)
            h = match(p.get(frame, []), j2) if p else None
            if h is not None:
                skel(img, h["kp2d"], BGR[model], 2)
            status.append(("MP" if model == "mediapipe" else "WiLoR") + (" ok" if h is not None else " MISS"))
        t = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], (300, 300), interpolation=cv2.INTER_AREA)
        label(t, f"{cond}: {name}"); label(t, " | ".join(status), y=290, scale=0.5)
        tiles.append(t)
    cv2.imwrite(str(ROOT / "report" / out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def real_detection(out="fig_real_detection.jpg"):
    sys.path.insert(0, str(ROOT / "eval"))
    import roboflow_bench as RB
    cfg = yaml.safe_load(open(ROOT / "configs/roboflow_bench.yaml"))
    D = ROOT / cfg["dataset"]["root"] / cfg["dataset"]["split"]
    coco = json.load(open(D / "_annotations.coco.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    excl = set(cfg["exclude_glove_unworn"]) | set(cfg["exclude_glove_unsure"])
    hits, misses = [], []
    for im in sorted(coco["images"], key=lambda i: i["id"]):
        g = [a for a in coco["annotations"] if a["image_id"] == im["id"] and cats[a["category_id"]] == "glove"]
        if not g or im["id"] in excl:
            continue
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in g]
        stem = pathlib.Path(im["file_name"]).stem
        w = json.load(open(ROOT / f"data/cache/roboflow/{stem}.wilor.json"))["hands"]
        m = json.load(open(ROOT / f"data/cache/roboflow/{stem}.mediapipe.json"))["hands"]
        hit = any(RB.recall_hits(boxes, [RB.kp_box(h["kp2d"]) for h in w]))
        (hits if hit else misses).append((im, boxes, w, m))
    tiles = []
    for (im, boxes, w, m), ok in [(x, True) for x in hits[:4]] + [(x, False) for x in misses[:4]]:
        img = cv2.imread(str(D / im["file_name"]))
        for b in boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), BGR["gt"], 2)
        for h in m:
            skel(img, h["kp2d"], BGR["mediapipe"], 3)
        for h in w:
            skel(img, h["kp2d"], BGR["wilor"], 3)
        t = cv2.resize(img, (260, 260), interpolation=cv2.INTER_AREA)
        label(t, f"WiLoR {'hit' if ok else 'MISS'}  MP {'hit' if any(RB.recall_hits(boxes, [RB.kp_box(h['kp2d']) for h in m])) else 'miss'}", scale=0.5)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
    cv2.imwrite(str(ROOT / "report" / out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    pose_examples()
    real_detection()
    print("examples written")
