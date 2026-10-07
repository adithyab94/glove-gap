"""Real-glove check: detection recall on Roboflow 'Gloves and bare hands' test split (n=83 images).
Needs `make rf-infer` (WiLoR + MediaPipe on the test images). Usage: roboflow_bench.py
Recall at IoU >= 0.5 vs GT boxes, by class (glove / bare) x box-size tercile, Wilson 95% CI.
Ablation: MediaPipe run on WiLoR-detector crops, to split detection failure from landmark failure."""
import json, pathlib, sys, csv
import cv2, numpy as np, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infer"))
cfg = yaml.safe_load(open(ROOT / "configs/roboflow_bench.yaml"))
D = ROOT / cfg["dataset"]["root"] / cfg["dataset"]["split"]
PRED = ROOT / "data/cache/roboflow"
OUT = ROOT / "eval/out"; OUT.mkdir(parents=True, exist_ok=True)
IOU = cfg["match"]["iou"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def kp_box(k2, pad=0.10, W=640, H=640):
    k = np.asarray(k2); x0, y0 = k.min(0); x1, y1 = k.max(0); w, h = x1 - x0, y1 - y0
    return [max(0, x0 - pad * w), max(0, y0 - pad * h), min(W, x1 + pad * w), min(H, y1 + pad * h)]


def recall_hits(gts, preds):
    """Greedy one-to-one matching by IoU; returns list of bools per GT box."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gts) for j, p in enumerate(preds)), reverse=True)
    used_g, used_p, hit = set(), set(), [False] * len(gts)
    for s, i, j in pairs:
        if s < IOU:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); hit[i] = True
    return hit


def mp_on_crops(img, wboxes, scale):
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    out = []
    H, W = img.shape[:2]
    for b in wboxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = scale * max(b[2] - b[0], b[3] - b[1]) / 2
        x0, y0, x1, y1 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        res = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for lm in res.multi_hand_landmarks or []:
            ch, cw = crop.shape[:2]
            out.append(kp_box([[x0 + l.x * cw, y0 + l.y * ch] for l in lm.landmark]))
    hands.close()
    return out


if __name__ == "__main__":
    coco = json.load(open(D / "_annotations.coco.json"))
    cats = {c["id"]: cfg["dataset"]["classes"].get(c["name"]) for c in coco["categories"]}
    excl = set(cfg["exclude_glove_unworn"]) | set(cfg["exclude_glove_unsure"])
    gts = []   # (image_id, file, class, box, sqrt_area, excluded)
    for a in coco["annotations"]:
        cl = cats[a["category_id"]]
        if cl is None:
            continue
        x, y, w, h = a["bbox"]
        im = next(i for i in coco["images"] if i["id"] == a["image_id"])
        gts.append((a["image_id"], im["file_name"], cl, [x, y, x + w, y + h], np.sqrt(w * h),
                    cl == "glove" and a["image_id"] in excl))
    main = [g for g in gts if not g[5]]
    edges = np.percentile([g[4] for g in main], [100 / 3, 200 / 3])
    tercile = lambda s: ["S", "M", "L"][int(np.searchsorted(edges, s))]

    by_img = {}
    for i, g in enumerate(gts):
        by_img.setdefault(g[1], []).append(i)
    hits = {k: [None] * len(gts) for k in ["mediapipe", "wilor", "wilor_yolo_box", "mp_on_wilor_crops"]}
    for fname, idx in by_img.items():
        img = cv2.imread(str(D / fname)); stem = pathlib.Path(fname).stem
        w = json.load(open(PRED / f"{stem}.wilor.json"))["hands"]
        m = json.load(open(PRED / f"{stem}.mediapipe.json"))["hands"]
        preds = {"mediapipe": [kp_box(h["kp2d"]) for h in m],
                 "wilor": [kp_box(h["kp2d"]) for h in w],
                 "wilor_yolo_box": [h["bbox"] for h in w],
                 "mp_on_wilor_crops": mp_on_crops(img, [h["bbox"] for h in w], cfg["ablation"]["crop_scale"])}
        for k, p in preds.items():
            for i, h in zip(idx, recall_hits([gts[i][3] for i in idx], p)):
                hits[k][i] = h

    rows = []
    for subset, sel in (("main_worn", lambda g: not g[5]), ("all_glove_boxes", lambda g: True)):
        for k in hits:
            for cl in ("bare", "glove"):
                for t in ("all", "S", "M", "L"):
                    ii = [i for i, g in enumerate(gts) if sel(g) and g[2] == cl and (t == "all" or tercile(g[4]) == t)]
                    n = len(ii); kk = sum(hits[k][i] for i in ii); p, lo, hi = wilson(kk, n)
                    rows.append({"subset": subset, "method": k, "class": cl, "size": t, "n": n, "hits": kk,
                                 "recall": round(p, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    with open(OUT / "roboflow_recall.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("tercile edges (sqrt area px):", edges.round(1))
    for r in rows:
        if r["subset"] == "main_worn" and r["size"] == "all":
            print(f"{r['method']:18s} {r['class']:5s} recall {r['recall']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  n={r['n']}")

    # Size confound check (P4): worn-glove boxes skew large. Size-standardise glove recall to the
    # bare-box tercile mix (direct standardisation), per detector, main_worn subset.
    std_rows = []
    for k in ["mediapipe", "wilor"]:
        ws, acc = 0, 0.0
        for t in ("S", "M", "L"):
            nb = sum(1 for g in gts if not g[5] and g[2] == "bare" and tercile(g[4]) == t)
            ig = [i for i, g in enumerate(gts) if not g[5] and g[2] == "glove" and tercile(g[4]) == t]
            if ig:
                acc += nb * np.mean([hits[k][i] for i in ig]); ws += nb
        std_rows.append({"method": k, "glove_recall_size_std": round(acc / ws, 3)})
        print(f"{k}: glove recall standardised to bare size mix = {acc / ws:.3f}")
    with open(OUT / "roboflow_size_std.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(std_rows[0])); wr.writeheader(); wr.writerows(std_rows)

    # Match-rule sensitivity (README examples showed WiLoR skeletons on gloves that fail IoU>=0.5
    # because glove boxes include the cuff/wrist): recall under IoU>=0.5, IoU>=0.3, and
    # ">=50% of the predicted box inside the GT box".
    def inside(g, p):
        i = max(0, min(g[2], p[2]) - max(g[0], p[0])) * max(0, min(g[3], p[3]) - max(g[1], p[1]))
        return i / max((p[2] - p[0]) * (p[3] - p[1]), 1e-9)
    sens = []
    for k in ["mediapipe", "wilor"]:
        for cl in ("bare", "glove"):
            ii = [i for i, g in enumerate(gts) if not g[5] and g[2] == cl]
            v = {"iou50": [], "iou30": [], "inside50": []}
            for i in ii:
                g = gts[i]; stem = pathlib.Path(g[1]).stem
                ps = [kp_box(h["kp2d"]) for h in json.load(open(PRED / f"{stem}.{k}.json"))["hands"]]
                v["iou50"].append(max([iou(g[3], p) for p in ps], default=0) >= 0.5)
                v["iou30"].append(max([iou(g[3], p) for p in ps], default=0) >= 0.3)
                v["inside50"].append(max([inside(g[3], p) for p in ps], default=0) >= 0.5)
            for rule, hv in v.items():
                p_, lo, hi = wilson(sum(hv), len(hv))
                sens.append({"method": k, "class": cl, "rule": rule, "n": len(hv), "recall": round(p_, 3),
                             "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    with open(OUT / "roboflow_match_sensitivity.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(sens[0])); wr.writeheader(); wr.writerows(sens)
    for r in sens:
        if r["class"] == "glove":
            print(f"match sensitivity {r['method']:9s} glove {r['rule']:8s} {r['recall']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
