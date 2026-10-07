"""GT loading, calibration, hand matching and geometry shared by eval scripts."""
import csv, json, pathlib
from functools import lru_cache
import cv2
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
DEX = ROOT / CFG["dexycb"]["root"]
SUBJ = DEX / CFG["dexycb"]["subject"]
PAIR = CFG["dexycb"]["pair"]
FPS = CFG["dexycb"]["fps_src"]
MIN_MASK_PX = 500          # a view "sees" the hand if its GT mask has >= 500 px
DET_IOU = 0.3              # detected = best predicted hand's keypoint box IoU with GT keypoint box


class _L(yaml.SafeLoader):
    pass


_L.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: l.construct_sequence(n))


@lru_cache(None)
def intrinsics(cam):
    c = yaml.load(open(DEX / "calibration/intrinsics" / f"{cam}_640x480.yml"), Loader=_L)["color"]
    return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float64)


@lru_cache(None)
def extrinsics(name):
    """{cam: 4x4 cam->world(master)}"""
    e = yaml.load(open(DEX / "calibration" / f"extrinsics_{name}" / "extrinsics.yml"), Loader=_L)["extrinsics"]
    out = {}
    for k, v in e.items():
        if len(v) == 12:
            T = np.eye(4); T[:3] = np.array(v).reshape(3, 4); out[k] = T
    return out


@lru_cache(None)
def seq_meta(seq):
    return yaml.safe_load(open(SUBJ / seq / "meta.yml"))


def world_from_cam(seq, cam):
    return extrinsics(seq_meta(seq)["extrinsics"])[cam]


@lru_cache(maxsize=4096)
def gt(seq, cam, frame):
    lab = np.load(SUBJ / seq / cam / f"labels_{frame:06d}.npz")
    j3, j2 = lab["joint_3d"][0].astype(np.float64), lab["joint_2d"][0].astype(np.float64)
    vis = bool((j2 >= 0).all() and (j3 != -1).all() and (lab["seg"] == 255).sum() >= MIN_MASK_PX)
    return j3, j2, vis


def depth(seq, cam, frame):
    return cv2.imread(str(SUBJ / seq / cam / f"aligned_depth_to_color_{frame:06d}.png"), cv2.IMREAD_ANYDEPTH)


def load_pred(model, cond, seq, cam):
    p = ROOT / "data/cache/pred" / model / cond / seq / f"{cam}.jsonl"
    if not p.exists():
        return None
    return {r["frame"]: r["hands"] for r in map(json.loads, open(p))}


def box(k2):
    k2 = np.asarray(k2)
    return np.r_[k2.min(0), k2.max(0)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(hands, gt2d):
    """Best predicted hand by keypoint-box IoU with GT; None if below DET_IOU.
    Same rule for every model, so detection rates are comparable."""
    if not hands:
        return None
    g = box(gt2d)
    s = [iou(box(h["kp2d"]), g) for h in hands]
    i = int(np.argmax(s))
    return hands[i] if s[i] >= DET_IOU else None


def procrustes(p, g):
    """Similarity-align p to g (both Nx3) -> aligned p."""
    mp, mg = p.mean(0), g.mean(0)
    P, G = p - mp, g - mg
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (P ** 2).sum()
    return s * P @ R + mg


def solve_translation(rel, kp2d, K):
    """Root-relative 3D (m) + 2D (px) + true K -> t so that K(rel + t) projects onto kp2d (LS)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A, b = [], []
    for (X, Y, Z), (u, v) in zip(rel, kp2d):
        A.append([fx, 0, cx - u]); b.append((u - cx) * Z - fx * X)
        A.append([0, fy, cy - v]); b.append((v - cy) * Z - fy * Y)
    t, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return t


def triangulate(kp2ds, Ps):
    """DLT per joint. kp2ds: list of (21,2); Ps: list of 3x4 world->pixel. -> (21,3) world."""
    out = np.zeros((21, 3))
    for j in range(21):
        A = []
        for k, P in zip(kp2ds, Ps):
            u, v = k[j]
            A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1]; out[j] = X[:3] / X[3]
    return out


def proj_matrix(seq, cam):
    T = np.linalg.inv(world_from_cam(seq, cam))  # world -> cam
    return intrinsics(cam) @ T[:3]


def reproj_err(Xw, kp2d, P):
    x = (P @ np.c_[Xw, np.ones(len(Xw))].T).T
    return np.linalg.norm(x[:, :2] / x[:, 2:3] - kp2d, axis=1)


def sequences(include_excluded=False):
    seqs = sorted({r["seq"] for r in csv.DictReader(open(ROOT / "data/meta.csv"))})
    if include_excluded:
        return seqs
    return [s for s in seqs if s not in CFG["analysis"]["exclude_main"]]


def conditions():
    import sys
    sys.path.insert(0, str(ROOT / "capture"))
    from synth_gloves import conditions as c
    return [n for n, _, _ in c(CFG["gloves"])]
