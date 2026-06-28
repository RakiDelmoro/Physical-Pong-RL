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
from .tracker import ObjectTracker, BehaviorUtility
from .model import ClassModel


class Perception:
    # identity gate: a class's action-effect (normalized Δcy range) must clear
    # this to be labeled "controlled". Low enough to survive the mild dilution
    # that happens when a new controlled object binds to the class and its data
    # (different cy range / saturation) joins the shared W.
    ACTION_EFFECT_GATE = 0.005

    def __init__(self, frame_h, frame_w, delay=3, num_actions=3,
                 n_freq=2, cheb_degree=3, forgetting=0.999, z_thresh=None,
                 use_utility=True, utility_retire=0.0, utility_warmup=5,
                 top_k=6, permissive_z=1.5, velocity_hint_fn=None):
        self.frame_h = int(frame_h)
        self.frame_w = int(frame_w)
        # Seed mode (Move 1): the DEFAULT seed is top-K (keep the K most salient
        # regions per frame -- rank-based, brightness-invariant, no knife-edge).
        # This is what the rig uses; there is NO fallback to a fixed z-thresh by
        # default. top-K caps the candidate flood so faint real paddles compete
        # on equal footing with glare for the K slots (no threshold to lose
        # them behind); move 2 retires the salient junk (glare) downstream.
        # z_thresh remains available ONLY as an explicit opt-in (top_k=None,
        # z_thresh=set) for legacy/experimentation -- not the default path.
        self._top_k = top_k
        self._permissive_z = permissive_z
        self._z_thresh = z_thresh
        # MODEL-ASSISTED COAST: an optional callback the tracker calls when a
        # track coasts (occluded at a contact). velocity_hint_fn(track_dict)
        # -> (vx, vy) in PIXEL units, or None (fall back to freeze). Wired by
        # the agent loop to the world model -- keeps perception decoupled from
        # the world model module (no import). None = the legacy freeze.
        self._velocity_hint_fn = velocity_hint_fn
        self.utility = (BehaviorUtility(retire_below=utility_retire,
                                        warmup=utility_warmup)
                        if use_utility else None)
        if self.utility is not None:
            self.utility.set_neutral(num_actions - 1)  # STAY = last action
        self.tracker = ObjectTracker(utility=self.utility)
        self.classmodel = ClassModel(delay=delay, num_actions=num_actions,
                                     n_freq=n_freq, cheb_degree=cheb_degree,
                                     forgetting=forgetting)
        self._last_action = None
        self._step = 0

    def reset(self):
        self.tracker.reset()
        if self.utility is not None:
            self.utility = BehaviorUtility(
                retire_below=self.utility.retire_below,
                warmup=self.utility.warmup)
            self.utility.set_neutral(self.classmodel.res.num_actions - 1)
            self.tracker = ObjectTracker(utility=self.utility)
        self.classmodel = ClassModel(
            delay=self.classmodel.res.delay,
            num_actions=self.classmodel.res.num_actions,
            n_freq=self.classmodel.res.n_freq,
            cheb_degree=self.classmodel.res.cheb_degree,
            forgetting=self.classmodel.forgetting)
        self._last_action = None
        self._step = 0

    def step(self, gray, action):
        # Default seed = top-K (rank-based, brightness-invariant). The legacy
        # fixed-z-thresh path is reached ONLY if the caller explicitly set
        # top_k=None and z_thresh=<value>; otherwise top-K is used. No silent
        # fallback to a default z-thresh.
        if self._top_k is not None:
            cands = propose_regions(gray, top_k=self._top_k,
                                    permissive_z=self._permissive_z)
        elif self._z_thresh is not None:
            cands = propose_regions(gray, z_thresh=self._z_thresh)
        else:
            # top_k=None and no z_thresh given -> still use top-K=6 rather than
            # the bare propose_regions default, so the no-fallback invariant
            # holds everywhere.
            cands = propose_regions(gray, top_k=6, permissive_z=self._permissive_z)
        tracks = self.tracker.update(cands, self._step,
                                     action=self._last_action,
                                     velocity_hint_fn=self._velocity_hint_fn)

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
        # Expose each track's behavior CLASS id (perception's 'objects that
        # move the same way' label) so downstream consumers (the world model)
        # can key a stable slot by CLASS instead of by the raw track id, which
        # churns for fast bouncy objects. None while the object is still in
        # its pre-binding 'watch and see' phase (perception deliberately
        # delays commitment to GATE_OBS observations).
        for t in tracks:
            obj = self.classmodel.objects.get(t["id"])
            t["class_id"] = (obj.bound_class.id
                              if obj is not None and obj.bound_class is not None
                              else None)
        return tracks, controlled

    def diagnostics(self):
        return self.classmodel.diagnostics()
