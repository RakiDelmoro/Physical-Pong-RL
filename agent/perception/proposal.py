"""
The SCAFFOLD (part 1) — the proposal SEED under the object tracker.

Appearance is demoted from identity to seed. This module proposes candidate
regions that "stand out from their LOCAL surround." It is deliberately:

  - POLARITY-AGNOSTIC    : bright-on-dark OR dark-on-bright both fire.
  - SCALE-RELATIVE       : a pixel is compared to its local neighborhood, not
                           to a global threshold.
  - BRIGHTNESS-INVARIANT : a global rescale g -> k*g leaves the z-score
                           unchanged (local mean and local std both scale by k).
                           A fixed gray threshold would shift by k and break;
                           this does not.
  - MULTI-SCALE          : several box-filter radii so both small (ball) and
                           elongated (paddle) objects fire on the same frame.

It says NOTHING about what a region IS. It hands candidates to the tracker,
whose persistence/cohesion/continuity rules decide what is a real object.
Appearance never decides identity.
"""

import cv2
import numpy as np


def saliency_zscore(gray, radii=(8, 16, 24), eps=1e-3):
    """
    Multi-scale local-contrast z-score:
        z(x) = |g(x) - mean_local(x)| / (std_local(x) + eps)
    taken as the max over a few box-filter radii. Lighting-robust by
    construction (see module docstring).
    """
    g = gray.astype(np.float32)
    z = np.zeros_like(g)
    for r in radii:
        k = 2 * r + 1
        mu = cv2.boxFilter(g, cv2.CV_32F, (k, k))
        mu2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
        var = np.maximum(mu2 - mu * mu, 0.0)
        sigma = np.sqrt(var)
        z = np.maximum(z, np.abs(g - mu) / (sigma + eps))
    return z


def propose_regions(gray, z_thresh=2.2, min_area=6, radii=(8, 16, 24),
                    close_ksize=7, open_ksize=3):
    """
    Candidate regions from the seed. Returns a list of dicts:
        {cx, cy, w, h, area, aspect, extent}
    where extent = area/(w*h) is "filled-ness". CANDIDATES only; the tracker
    decides which become objects.
    """
    z = saliency_zscore(gray, radii=radii)
    mask = (z > z_thresh).astype(np.uint8) * 255
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):  # 0 = background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = cents[i]
        aspect = (w / float(h)) if h > 0 else 1.0
        extent = (area / float(w * h)) if (w * h) > 0 else 0.0
        out.append({"cx": float(cx), "cy": float(cy), "w": float(w),
                    "h": float(h), "area": float(area),
                    "aspect": aspect, "extent": extent})
    return out
