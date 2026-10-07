"""HaMeR runner -> standardized records. Boxes/handedness come from cached WiLoR detector output
(avoids detectron2/ViTPose deps), so HaMeR detection rate == WiLoR's by construction.
Smoke:  run_hamer.py IMG_DIR WILOR_JSON_DIR OUT_DIR
Batch:  run_hamer.py data/cache/frames.csv          (pair cams only; needs WiLoR batch first)
Run with cwd = third_party/hamer (relative _DATA paths)."""
import sys, json, pathlib, cv2, torch
import numpy as np
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import cam_crop_to_full
from schema import record, dump, groups, run_group, pred_path

torch.manual_seed(0)
dev = torch.device("cuda")
model, cfg = load_hamer(DEFAULT_CHECKPOINT)
model = model.to(dev).eval()


def hamer_on_boxes(img, dets):
    if not dets:
        return []
    boxes = np.array([d["bbox"] for d in dets])
    right = np.array([d["handedness"] == "Right" for d in dets])
    dl = torch.utils.data.DataLoader(ViTDetDataset(cfg, img, boxes, right), batch_size=8)
    hands = []
    for batch in dl:
        batch = recursive_to(batch, dev)
        with torch.no_grad():
            out = model(batch)
        mult = 2 * batch["right"] - 1
        cam = out["pred_cam"]; cam[:, 1] = mult * cam[:, 1]
        img_size = batch["img_size"].float()
        f = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        t_full = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(), img_size, f)
        k3 = out["pred_keypoints_3d"].clone(); k3[:, :, 0] *= mult[:, None]
        k3c = (k3 + t_full[:, None]).cpu().numpy()          # camera frame, metres (assumed focal)
        k2 = (k3c[..., :2] / k3c[..., 2:3]) * f.item() + (img_size.cpu().numpy()[:, None] / 2)
        k3r = k3.cpu().numpy()
        for i in range(k3c.shape[0]):
            hands.append({"handedness": "Right" if batch["right"][i] > 0.5 else "Left", "conf": None,
                          "bbox": boxes[len(hands)].tolist(), "kp2d": k2[i].tolist(),
                          "kp3d_rel": (k3r[i] - k3r[i, 0]).tolist(), "kp3d": k3c[i].tolist()})
    return hands


if __name__ == "__main__":
    if sys.argv[1].endswith(".csv"):
        g = groups(sys.argv[1], pair_only=True); n = 0
        for i, (key, items) in enumerate(sorted(g.items())):
            dets = {}
            for line in open(pred_path("wilor", *key)):
                r = json.loads(line); dets[r["frame"]] = r["hands"]
            n += run_group("hamer", key, items, lambda img, k, f: hamer_on_boxes(img, dets[f]), cv2.imread)
            if i % 50 == 0:
                print(f"hamer group {i}/{len(g)}", flush=True)
        print(f"hamer: {n} new frames, {len(g)} groups")
    else:
        img_dir, det_dir, out_dir = map(pathlib.Path, sys.argv[1:4])
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(img_dir.glob("*")):
            img = cv2.imread(str(p)); dj = det_dir / f"{p.stem}.wilor.json"
            if img is None or not dj.exists():
                continue
            hands = hamer_on_boxes(img, json.load(open(dj))["hands"])
            dump(record("hamer", p.name, hands), out_dir / f"{p.stem}.hamer.json")
            print(p.name, len(hands), "hands")
