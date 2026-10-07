"""WiLoR(-mini) runner -> standardized records.
Smoke:  run_wilor.py IMG_DIR OUT_DIR
Batch:  run_wilor.py data/cache/frames.csv
kp3d = camera frame in metres under WiLoR's assumed focal (5000/256 * max(H, W)), not the true
intrinsics; eval/ re-solves translation with true K. kp3d_rel = root-relative MANO joints."""
import sys, pathlib, cv2, torch
import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from schema import smoke, groups, run_group

ROOT = pathlib.Path(__file__).resolve().parents[1]
torch.manual_seed(0)
pipe = WiLorHandPose3dEstimationPipeline(
    device=torch.device("cuda"), dtype=torch.float16,
    wilor_pretrained_dir=str(ROOT / "third_party" / "wilor"), verbose=False)


def predict(img, key, frame):
    outs = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hands = []
    for o in outs:
        w = o.get("wilor_preds")
        if w is None:
            continue
        k3 = np.asarray(w["pred_keypoints_3d"])[0]
        hands.append({"handedness": "Right" if o["is_right"] else "Left",
                      "conf": None,  # detector conf not exposed by wrapper
                      "bbox": o["hand_bbox"],
                      "kp2d": np.asarray(w["pred_keypoints_2d"])[0].tolist(),
                      "kp3d_rel": (k3 - k3[0]).tolist(),
                      "kp3d": (k3 + np.asarray(w["pred_cam_t_full"])[0]).tolist()})
    return hands


if __name__ == "__main__":
    if sys.argv[1].endswith(".csv"):
        g = groups(sys.argv[1]); n = 0
        for i, (key, items) in enumerate(sorted(g.items())):
            n += run_group("wilor", key, items, predict, cv2.imread)
            if i % 50 == 0:
                print(f"wilor group {i}/{len(g)}", flush=True)
        print(f"wilor: {n} new frames, {len(g)} groups")
    else:
        smoke("wilor", sys.argv[1], sys.argv[2], predict, cv2.imread)
