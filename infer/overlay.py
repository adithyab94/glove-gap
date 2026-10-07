"""Draw standardized 2D keypoints from all models side by side. Usage: overlay.py IMG_DIR JSON_DIR OUT_JPG"""
import sys, json, pathlib, cv2, numpy as np

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
MODELS = ["mediapipe", "wilor", "hamer"]
img_dir, js_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
rows = []
for p in sorted(img_dir.glob("*.jpg")):
    img = cv2.imread(str(p)); tiles = []
    for m in MODELS:
        t = img.copy(); f = js_dir / f"{p.stem}.{m}.json"
        hands = json.load(open(f))["hands"] if f.exists() else []
        for h in hands:
            k = np.asarray(h["kp2d"]).astype(int)
            for a, b in BONES:
                cv2.line(t, tuple(k[a]), tuple(k[b]), (0, 255, 0), max(2, img.shape[1] // 300))
        t = cv2.resize(t, (400, int(400 * img.shape[0] / img.shape[1])))
        t = cv2.copyMakeBorder(t, 0, 600 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) if t.shape[0] < 600 else t[:600]
        cv2.putText(t, f"{m}: {len(hands)}", (8, 30), 0, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows))
