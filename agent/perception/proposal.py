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
                    close_ksize=7, open_ksize=3, top_k=None,
                    permissive_z=1.5):
    """
    Candidate regions from the seed. Returns a list of dicts:
        {cx, cy, w, h, area, aspect, extent, salience}
    where extent = area/(w*h) is "filled-ness". CANDIDATES only; the tracker
    decides which become objects.

    Two seed modes (Move 1 = top-K):
      - top_k=None (default): the legacy fixed-z-thresh seed. Keep every
        connected component whose z > z_thresh. This is the knife-edge: too
        high and faint real objects (a dim paddle) are lost; too low and
        junk floods in.
      - top_k=K (Move 1): keep the K most salient regions per frame, whatever
        the absolute z. We threshold at a low PERMISSIVE z (just to form
        connected components), then rank regions by mean z over their pixels
        and keep the top K. The RANK is brightness-invariant (a g->k*g shift
        leaves z unchanged), so 'the K most noticeable things' is stable
        across lighting -- no knife-edge. K is a far gentler knob than an
        absolute z threshold ('about 6-8 noticeable things' is roughly right
        in any lighting). Glare will still be in the top-K (it is salient);
        Move 2 (behavioral utility) retires it downstream. So top-K + move 2
        is the combination that makes z-thresh stop mattering.
    """
    z = saliency_zscore(gray, radii=radii)
    thresh = permissive_z if top_k is not None else z_thresh
    mask = (z > thresh).astype(np.uint8) * 255
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)

    # Build candidate regions from connected components, each tagged with its
    # mean z-score (salience). In top-K mode we rank by salience and keep the
    # K most salient; in legacy mode we keep them all (the fixed z-thresh
    # already filtered). One labeling pass, shared by both paths.
    n_lbl, lbl = cv2.connectedComponents(mask, 8)
    out = []
    for i in range(1, n_lbl):  # 0 = background
        ys, xs = np.where(lbl == i)
        area = float(len(xs))
        if area < min_area:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        w0 = int(xs.max()) - x0 + 1
        h0 = int(ys.max()) - y0 + 1
        cx, cy = float(xs.mean()), float(ys.mean())
        aspect = (w0 / float(h0)) if h0 > 0 else 1.0
        extent = (area / float(w0 * h0)) if (w0 * h0) > 0 else 0.0
        sal = float(z[ys, xs].mean())
        out.append({"cx": cx, "cy": cy, "w": float(w0), "h": float(h0),
                    "area": area, "aspect": aspect, "extent": extent,
                    "salience": sal})
    if top_k is not None and out:
        out.sort(key=lambda r: r["salience"], reverse=True)
        return out[:top_k]
    return out
