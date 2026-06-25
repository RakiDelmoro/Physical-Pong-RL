"""
Perception — the top level: the SCAFFOLD feeds the MODEL.

step(gray, action) -> (tracks, controlled_id)

  gray   : HxW uint8 grayscale frame from the camera.
  action : int, the action taken AT this frame (it leads to the next frame;
           we store it and use it to train the model on the NEXT step, when
           we see where each object actually went).

  tracks : list of confirmed tracked objects (stable ids, no names).
  controlled_id : the track id the agent controls (identity by behavior),
           or None if no class is confident yet.

The scaffold (proposal + tracker) finds objects in pixel coords. The model
(classmodel) learns each object's dynamics in NORMALIZED coords and discovers
behavior classes. Identity is read off the model: the object bound to the
class whose predicted position varies most with the action is the controlled
object. Lighting can't break that — behavior is lighting-invariant.
"""

import numpy as np

from .proposal import propose_regions
from .tracker import ObjectTracker
from .model import ClassModel


class Perception:
    # identity gate: a class's action-effect (normalized Δcy range) must clear
    # this to be labeled "controlled". Low enough to survive the mild dilution
    # that happens when a new controlled object binds to the class and its data
    # (different cy range / saturation) joins the shared W.
    ACTION_EFFECT_GATE = 0.005

    def __init__(self, frame_h, frame_w, delay=3, num_actions=3,
                 n_freq=2, cheb_degree=3, forgetting=0.999, z_thresh=None):
        self.frame_h = int(frame_h)
        self.frame_w = int(frame_w)
        # z_thresh: per-frame seed sensitivity. None = propose_regions default
        # (good for the clean synthetic scene). Real video is lower-contrast and
        # noisier (score text, dashed net, glare) so a higher value is needed
        # there; the caller tunes it for the rig.
        self._z_thresh = z_thresh
        self.tracker = ObjectTracker()
        self.classmodel = ClassModel(delay=delay, num_actions=num_actions,
                                     n_freq=n_freq, cheb_degree=cheb_degree,
                                     forgetting=forgetting)
        self._last_action = None
        self._step = 0

    def reset(self):
        self.tracker.reset()
        self.classmodel = ClassModel(
            delay=self.classmodel.res.delay,
            num_actions=self.classmodel.res.num_actions,
            n_freq=self.classmodel.res.n_freq,
            cheb_degree=self.classmodel.res.cheb_degree,
            forgetting=self.classmodel.forgetting)
        self._last_action = None
        self._step = 0

    def step(self, gray, action):
        cands = (propose_regions(gray) if self._z_thresh is None
                 else propose_regions(gray, z_thresh=self._z_thresh))
        tracks = self.tracker.update(cands, self._step)

        live = set()
        for t in tracks:
            tid = t["id"]
            live.add(tid)
            # train only on FRESH observations (skip coasted/fake positions).
            # the action used is the one sent on the PREVIOUS step, which is
            # the action that produced this new position.
            if t["missed"] == 0 and self._last_action is not None:
                pos = np.array([t["cx"] / self.frame_w, t["cy"] / self.frame_h])
                self.classmodel.observe(tid, pos, self._last_action)

        # retire objects whose tracks the scaffold has dropped
        for tid in list(self.classmodel.objects):
            if tid not in live:
                self.classmodel.retire(tid)

        self.classmodel.update()
        self._last_action = int(action)
        self._step += 1
        controlled = self.classmodel.controlled_object(self.ACTION_EFFECT_GATE)
        return tracks, controlled

    def diagnostics(self):
        return self.classmodel.diagnostics()
