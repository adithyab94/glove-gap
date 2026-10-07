"""MediaPipe Hands runner -> standardized records.
Smoke:  run_mediapipe.py IMG_DIR OUT_DIR
Batch:  run_mediapipe.py data/cache/frames.csv [N_PROCS]   (CPU, multi-process; pair cams only)
static_image_mode=True: per-frame detection, no tracker, same footing as WiLoR."""
import sys, cv2
from multiprocessing import Pool
from schema import smoke, groups, run_group

_hands = None


def _get():
    global _hands
    if _hands is None:
        import mediapipe as mp
        _hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                          min_detection_confidence=0.3)
    return _hands


def predict(img, key, frame):
    h, w = img.shape[:2]
    res = _get().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for lm, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
        out.append({"handedness": hd.classification[0].label,
                    "conf": hd.classification[0].score,
                    "kp2d": [[l.x * w, l.y * h] for l in lm.landmark],
                    "kp3d": [[l.x, l.y, l.z] for l in lm.landmark]})  # normalised, not metric
    return out


def _job(a):
    return run_group("mediapipe", a[0], a[1], predict, cv2.imread)


if __name__ == "__main__":
    if sys.argv[1].endswith(".csv"):
        g = groups(sys.argv[1], pair_only=True)  # 8-cam upper bound is WiLoR only
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 16) as pool:
            n = sum(pool.imap_unordered(_job, g.items()))
        print(f"mediapipe: {n} new frames, {len(g)} groups")
    else:
        smoke("mediapipe", sys.argv[1], sys.argv[2], predict, cv2.imread)
