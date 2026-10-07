"""Synthetic glove appearance applied inside a hand mask.

Shading is taken from the original image's luminance (so lighting and finger self-shadowing
survive), then the skin albedo is replaced by a glove colour, fine detail (creases, nails) is
low-passed, and an optional procedural texture is added. G3b additionally grows the hand
outline by k px in 2D; skeleton GT is untouched (real thick gloves grow the surface, not the
bones). Deterministic given (image, mask, spec, seed).
"""
import cv2
import numpy as np


def _shading(img, mask, blur, gamma):
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur > 0:
        # Low-pass only within the hand: normalised convolution so background doesn't bleed in.
        m = mask.astype(np.float32)
        lum = cv2.GaussianBlur(lum * m, (0, 0), blur) / np.maximum(cv2.GaussianBlur(m, (0, 0), blur), 1e-3)
    ref = np.median(lum[mask]) if mask.any() else 1.0
    s = np.clip(lum / max(ref, 1.0), 0, 2.5)
    return s ** gamma


def _texture(shape, kind, period, amp, rng):
    if kind is None:
        return np.ones(shape, np.float32)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, np.pi)
    u = xx * np.cos(th) + yy * np.sin(th)
    if kind == "rib":     # knit ribs + yarn noise
        t = np.sin(2 * np.pi * u / period)
        t += 0.5 * cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 0.7)
    elif kind == "quilt":  # stitched quilting lines on a padded surface
        v = -xx * np.sin(th) + yy * np.cos(th)
        t = -np.exp(-((u % period) ** 2) / 2.0) - np.exp(-((v % (2 * period)) ** 2) / 2.0)
        t += 0.3 * cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 1.5)
    else:
        raise ValueError(kind)
    t = t / (np.abs(t).max() + 1e-6)
    return 1.0 + amp * t


def grow_mask(mask, k):
    if k <= 0:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), ker).astype(bool)


def skin_rule(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)


def refine_mask(img, seg, joints2d=None, max_px=6, hull_pad=10):
    """DexYCB hand masks are rendered from the MANO fit and miss real skin by 2-5 px (see
    capture/mask_leak.py). Absorb skin-coloured background pixels within max_px of the mask,
    connected to it. Where the fit put the object in front of the hand, real skin is labelled
    as object: inside the (padded) convex hull of the GT 2D joints, skin-coloured object
    pixels are absorbed too."""
    mask = seg == 255
    skin = skin_rule(img)
    cand = grow_mask(mask, max_px) & (seg == 0) & skin
    if joints2d is not None and mask.sum() > 200:
        hull = np.zeros(seg.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.asarray(joints2d, np.int32).reshape(-1, 2)), 1)
        hull = grow_mask(hull.astype(bool), hull_pad)
        # Generic skin rule also passes wood/red/brown objects, so object pixels must match
        # this hand's own colour: Gaussian over (Cr, Cb) of eroded hand-mask pixels.
        crcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1:].astype(np.float32)
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        x = crcb[core if core.sum() > 100 else mask]
        mu, cov = x.mean(0), np.cov(x.T) + np.eye(2)
        dlt = crcb - mu
        md2 = np.einsum("hwi,ij,hwj->hw", dlt, np.linalg.inv(cov), dlt)
        cand |= hull & (seg != 0) & (seg != 255) & (md2 < 2.5 ** 2)
    n, lab = cv2.connectedComponents((mask | cand).astype(np.uint8), connectivity=8)
    keep = np.unique(lab[mask])
    return np.isin(lab, keep[keep > 0])


def apply_glove(img, mask, spec, seed=0, grow_px=0):
    """img: BGR uint8; mask: bool HxW hand pixels; spec: dict from configs/experiment.yaml."""
    if spec["kind"] == "bare" or not mask.any():
        return img.copy()
    rng = np.random.default_rng(seed)
    m = grow_mask(mask, grow_px)
    shade = _shading(img, mask, spec.get("detail_blur", 0.0), spec.get("shade_gamma", 1.0))
    if grow_px > 0:
        # Grown ring has no hand shading underneath: extend hand shading outward by inpainting.
        ring = (m & ~mask).astype(np.uint8)
        s8 = np.clip(shade * 100, 0, 255).astype(np.uint8)
        shade = cv2.inpaint(s8, ring, 3, cv2.INPAINT_TELEA).astype(np.float32) / 100
    tex = _texture(mask.shape, spec.get("tex"), spec.get("tex_period_px", 4), spec.get("tex_amp", 0.0), rng)
    rgb = np.asarray(spec["rgb"], np.float32)[::-1]  # -> BGR
    glove = np.clip(rgb[None, None] * (shade * tex)[..., None], 0, 255)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)[..., None]  # 1-px feathered edge
    out = alpha * glove + (1 - alpha) * img.astype(np.float32)
    return out.astype(np.uint8)


def conditions(cfg_gloves):
    """Expand config into concrete (name, spec, grow_px), e.g. G3b_k4."""
    out = []
    for name, spec in cfg_gloves.items():
        if "base" in spec:
            for k in spec["grow_px"]:
                out.append((f"{name}_k{k}", cfg_gloves[spec["base"]], k))
        else:
            out.append((name, spec, 0))
    return out
