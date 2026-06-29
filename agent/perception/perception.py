"""
Perception -- the SCAFFOLD only (option C: the labeler moved to the world model).

step(gray, action) -> (tracks, controlled_id)

  gray   : HxW uint8 grayscale frame from the camera.
  action : int, the action taken AT this frame.

  tracks : list of confirmed tracked objects (stable ids, no names).
  controlled_id : the track id the agent controls, or None. Under option C
           this comes from a `controlled_fn` callback (wired by the agent loop
           to the world model's `controlled_track` -- the dynamics model
           discovers which object the action moves). None if no callback or
           the model is cold. Perception itself no longer labels: it just finds
           and tracks objects (the scaffold). The labeling question ("which
           object am I controlling?") is a DYNAMICS question, answered by the
           world model -- the Alberta Plan's "dissolve the perception/labeling
           chicken-and-egg": one learned surface answers both "what next" and
           "what is each object".

The scaffold (proposal + tracker) finds objects in pixel coords and keeps
stable track ids across lighting shifts (brightness-invariant z-score seed +
behavioral-utility junk rejection). It is GENERAL (no Pong vocabulary) and
LIGHTING-INVARIANT by construction. The class-model labeler that used to live
here (with its fragile tower of ghosts / revival / binding / merging) is
DELETED -- the world model's slot assignment already keeps per-object identity
stable across re-detection, so the "carry the label across churn" heuristics
had no job once the labeler and the dynamics model became the same object.
"""

import numpy as np

from .proposal import propose_regions, propose_regions_conditioned
from .tracker import ObjectTracker, BehaviorUtility


class Perception:
    def __init__(self, frame_h, frame_w, delay=3, num_actions=3,
                 n_freq=2, cheb_degree=3, forgetting=0.999, z_thresh=None,
                 use_utility=True, utility_retire=0.0, utility_warmup=5,
                 top_k=6, permissive_z=1.5,
                 velocity_hint_fn=None, controlled_fn=None):
        # The ctor accepts several now-unused kwargs (delay, n_freq,
        # cheb_degree, forgetting) that used to configure the deleted class
        # model. Kept in the signature so existing callers don't break; they
        # are ignored. z_thresh / top_k / permissive_z still configure the
        # proposal seed (the scaffold's first move -- the conditioned-proposal
        # wiring is a separate, later step).
        self.frame_h = int(frame_h)
        self.frame_w = int(frame_w)
        self._top_k = top_k
        self._permissive_z = permissive_z
        self._z_thresh = z_thresh
        _ = delay, n_freq, cheb_degree, forgetting  # ignored (class-model era)
        # MODEL-ASSISTED COAST: an optional callback the tracker calls when a
        # track coasts (occluded at a contact). velocity_hint_fn(track_dict)
        # -> (vx, vy) in PIXEL units, or None (fall back to freeze). Wired by
        # the agent loop to the world model -- keeps perception decoupled.
        self._velocity_hint_fn = velocity_hint_fn
        # OPTION C: controlled-object discovery is delegated to a callback
        # (wired to the world model's controlled_track). Perception stays
        # decoupled from the world model (no import). None = no labeling
        # (pure scaffold; controlled is always None -- for scaffold-only tests).
        self._controlled_fn = controlled_fn
        self._num_actions = int(num_actions)
        self.utility = (BehaviorUtility(retire_below=utility_retire,
                                        warmup=utility_warmup)
                        if use_utility else None)
        if self.utility is not None:
            self.utility.set_neutral(self._num_actions - 1)  # STAY = last action
        self.tracker = ObjectTracker(utility=self.utility)
        self._last_action = None
        self._step = 0

    def reset(self):
        self.tracker.reset()
        if self.utility is not None:
            self.utility = BehaviorUtility(
                retire_below=self.utility.retire_below,
                warmup=self.utility.warmup)
            self.utility.set_neutral(self._num_actions - 1)
            self.tracker = ObjectTracker(utility=self.utility)
        self._last_action = None
        self._step = 0

    def step(self, gray, action):
        # TOP-DOWN when tracks exist: prediction-conditioned proposal. Each
        # existing track's predicted position carves out its OWN candidate
        # before connected components can merge anything, so identity is
        # carried top-down and a contact/merge can no longer destroy it (the
        # W3b blocker -- the ball losing its track at a paddle bounce).
        # BOTTOM-UP when cold (no tracks yet): legacy top-K proposal, same as
        # before -- cold start is unchanged. The bottom-up path is demoted to
        # 'new things only' once tracks exist (the residual of the conditioned
        # proposal). GENERAL -- no Pong vocabulary.
        preds = self.tracker.predictions()
        if preds:
            claimed, residual = propose_regions_conditioned(
                gray, preds, permissive_z=self._permissive_z)
            tracks = self.tracker.update_conditioned(
                claimed, residual, self._step,
                action=self._last_action,
                velocity_hint_fn=self._velocity_hint_fn)
        else:
            # cold start: legacy bottom-up proposal (top-K, brightness-
            # invariant). Reached only when the tracker has no tracks.
            if self._top_k is not None:
                cands = propose_regions(gray, top_k=self._top_k,
                                        permissive_z=self._permissive_z)
            elif self._z_thresh is not None:
                cands = propose_regions(gray, z_thresh=self._z_thresh)
            else:
                cands = propose_regions(gray, top_k=6,
                                        permissive_z=self._permissive_z)
            tracks = self.tracker.update(cands, self._step,
                                         action=self._last_action,
                                         velocity_hint_fn=self._velocity_hint_fn)
        self._last_action = int(action)
        self._step += 1
        # OPTION C: the controlled object is discovered by the world model
        # (the dynamics model -- "which object's motion correlates with my
        # action?"), not by a perception-internal labeler. If no callback is
        # wired, controlled is None (pure scaffold mode).
        controlled = None
        if self._controlled_fn is not None:
            try:
                controlled = self._controlled_fn(tracks)
            except Exception:
                controlled = None
        return tracks, controlled

    def diagnostics(self):
        return {"n_tracks": len(self.tracker._tracks)}
