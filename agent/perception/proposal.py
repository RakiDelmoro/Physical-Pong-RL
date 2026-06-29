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


# ====================================================================
# TOP-DOWN / PREDICTION-CONDITIONED proposal (the contact/identity fix).
# ====================================================================
# The legacy `propose_regions` above is BOTTOM-UP: it finds blobs by
# connected components, so any pixels that TOUCH become ONE blob. That rule
# destroys identity the moment two objects get close (ball touches paddle ->
# one merged blob -> the tracker loses the ball). Targeted fixes downstream
# (no-snap re-acquisition, model-assisted coast) all hit this same wall: the
# identity is gone before the tracker ever sees a candidate.
#
# The fix is TOP-DOWN (predictive coding, pushed into perception): we already
# KNOW where each existing track expects to be (its predicted position). So
# before connected components can merge anything, we assign each salient pixel
# to the NEAREST predicted track within that track's search radius. Each
# track's claimed pixels become its OWN candidate -- even if those pixels were
# part of a merged blob. Identity is carried TOP-DOWN (from the predictions),
# not derived BOTTOM-UP (from touching pixels), so a merge can no longer
# destroy it. Whatever salient pixels NO track explains -> connected
# components -> NEW-object candidates (the bottom-up path, demoted to 'new
# things only'). This is the BLENDED scaffold: top-down for known/occluded
# objects, bottom-up for surprises. GENERAL: no Pong vocabulary -- the rule is
# 'each pixel goes to the nearest predicted object within range; the rest is
# new', true in any world with moving objects.

def propose_regions_conditioned(gray, predictions, z_thresh=2.2, min_area=4,
                                 radii=(8, 16, 24), close_ksize=7,
                                 open_ksize=3, permissive_z=1.5):
    """Prediction-conditioned proposal. Top-down for known objects, bottom-up
    for new ones.

    `predictions` : list of (px, py, radius) -- each existing track's
        predicted pixel position and a search radius (how far the object may
        be from the prediction and still be claimed by that track). May be
        empty (cold start -> pure bottom-up, same as propose_regions).

    Returns (claimed, residual) where:
      claimed  : list of candidates, ONE per input prediction (in the same
                 order), or None for a prediction that found no salient
                 pixels in its radius (the object was not visible there --
                 the tracker will coast it). Each candidate is a dict with the
                 same shape as propose_regions' output, plus 'pred_idx'.
      residual : list of candidates from salient pixels NO track claimed
                 (new objects / surprises) -- the bottom-up path, demoted to
                 'new things only'.
    """
    z = saliency_zscore(gray, radii=radii)
    mask = (z > permissive_z).astype(np.uint8) * 255
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)

    preds = [(float(p[0]), float(p[1]), float(p[2])) for p in predictions]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [None] * len(preds), []

    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

    # ---- assign each salient pixel to the NEAREST predicted track within
    # that track's search radius (Voronoi conditioned on the predictions).
    # A pixel claimed by NO track is 'residual' (new). Duplicate fragments
    # (two tracks on one object) are cleaned up by the tracker's
    # _drop_duplicate_fragments AFTER this, with a size+direction guard that
    # protects a different small object (the ball) -- a proposal-stage
    # position-only rule cannot make that distinction and would suppress the
    # ball at a contact. ----
    owner = np.full(len(pts), -1, dtype=np.int64)   # -1 = residual / new
    if preds:
        best_d = np.full(len(pts), np.inf, dtype=np.float32)
        for pi, (px, py, rad) in enumerate(preds):
            d = np.sqrt((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
            take = (d < best_d) & (d <= rad)
            owner[take] = pi
            best_d[take] = d[take]

    def _cand_from_pixels(px, py):
        x0, y0 = int(px.min()), int(py.min())
        w0 = int(px.max()) - x0 + 1
        h0 = int(py.max()) - y0 + 1
        area = float(len(px))
        aspect = (w0 / float(h0)) if h0 > 0 else 1.0
        extent = (area / float(w0 * h0)) if (w0 * h0) > 0 else 0.0
        return {"cx": float(px.mean()), "cy": float(py.mean()),
                "w": float(w0), "h": float(h0), "area": area,
                "aspect": aspect, "extent": extent,
                "salience": float(z[py, px].mean())}

    # ---- per-prediction claimed candidates (one per track, in order) ----
    claimed = [None] * len(preds)
    for pi in range(len(preds)):
        sel = owner == pi
        if sel.sum() < min_area:
            continue   # nothing (or too little) at this prediction -> coast
        c = _cand_from_pixels(xs[sel], ys[sel])
        c["pred_idx"] = pi
        claimed[pi] = c

    # ---- residual: salient pixels no track claimed -> new objects (bottom-
    # up connected components, demoted to 'new things only') ----
    residual = []
    res_sel = owner == -1
    if res_sel.any():
        rx, ry = xs[res_sel], ys[res_sel]
        # label the residual pixels (they are a subset of the mask; build a
        # sub-mask and run connected components on it).
        sub = np.zeros_like(mask)
        sub[ry, rx] = 255
        n_lbl, lbl = cv2.connectedComponents(sub, 8)
        for i in range(1, n_lbl):
            yy, xx = np.where(lbl == i)
            if len(xx) < min_area:
                continue
            c = _cand_from_pixels(xx, yy)
            residual.append(c)
    return claimed, residual
