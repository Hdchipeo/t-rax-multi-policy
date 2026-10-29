"""Episode-local goals and target memory, independent of the motor controller.

Target estimates are supplied by perception. The MuJoCo adapter currently supplies
synthetic estimates; they must be replaced by measured estimates on hardware.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


class GoalType(IntEnum):
    REST = 0
    INSPECT = 1
    TRACK = 2
    SOCIAL = 3
    EXPLORE = 4
    ORIENT = 5


class GoalPhase(IntEnum):
    ORIENT = 0
    APPROACH = 1
    OBSERVE = 2
    WAIT = 3


@dataclass
class TargetEstimate:
    target_id: str
    kind: str
    position: np.ndarray
    distance: float
    bearing: float  # body-relative; camera error = bearing + head pan
    visible: bool
    speed: float = 0.0
    confidence: float = 0.9
    gaze_error: float = 0.0


@dataclass
class Goal:
    kind: GoalType
    target_id: Optional[str] = None
    phase: GoalPhase = GoalPhase.ORIENT
    elapsed: float = 0.0
    phase_elapsed: float = 0.0
    aligned_time: float = 0.0
    lost_time: float = 0.0
    progress: float = 0.0
    initial_distance: float = 0.0
    previous_distance: float = 0.0
    deadline: float = 18.0
    signal_started: bool = False
    responded: bool = False
    look_away_sign: float = 1.0


class GoalManager:
    def __init__(self, rest_seconds=3.0, inspection_seconds=2.0,
                 memory_cooldown_seconds=30.0):
        self.rest_seconds = rest_seconds
        self.inspection_seconds = inspection_seconds
        self.memory_cooldown_seconds = memory_cooldown_seconds
        if min(rest_seconds, inspection_seconds, memory_cooldown_seconds) <= 0:
            raise ValueError("Goal durations must be positive")
        self.reset()

    def reset(self):
        self.time = 0.0
        self.goal = Goal(GoalType.REST, phase=GoalPhase.WAIT, deadline=self.rest_seconds)
        self.memory = {}
        self.completed = self.failed = self.interrupted = 0
        self.last_result = "INITIAL_REST"
        self.completion_event = self.failure_event = False
        self.progress_delta = 0.0
        self.next_explore_time = 0.0

    def _phase(self, phase):
        self.goal.phase = phase
        self.goal.phase_elapsed = self.goal.aligned_time = 0.0

    def _finish(self, result, target=None):
        if self.goal.kind == GoalType.EXPLORE:
            self.next_explore_time = self.time + (15.0 if result == "SUCCESS" else 30.0)
        if result == "SUCCESS":
            self.completed += 1
            self.completion_event = True
        elif result == "SAFETY_INTERRUPTION":
            self.interrupted += 1
        else:
            self.failed += 1
            self.failure_event = True
        if target is not None:
            # Failed attempts also get a cooldown, avoiding endless retries.
            cooldown = self.memory_cooldown_seconds if result == "SUCCESS" else 8.0
            self.memory[target.target_id] = (self.time + cooldown, target.position.copy(), target.speed)
        self.last_result = result
        self.goal = Goal(GoalType.REST, target.target_id if target else None,
                         phase=GoalPhase.WAIT, deadline=self.rest_seconds,
                         look_away_sign=-1.0 if target and target.bearing > 0 else 1.0)

    def _available(self, target):
        remembered = self.memory.get(target.target_id)
        return (target.confidence >= .5 and target.distance < 3.5 and
                (remembered is None or self.time >= remembered[0] or
                 np.linalg.norm(target.position - remembered[1]) > .25 or
                 target.speed > .15 and remembered[2] <= .05))

    def step(self, dt, targets, *, scenario, curiosity, social_need, readiness,
             safety_active=False, acoustic=None):
        self.time += dt
        self.completion_event = self.failure_event = False
        self.progress_delta = 0.0
        g = self.goal
        g.elapsed += dt
        g.phase_elapsed += dt
        target = next((t for t in targets if t.target_id == g.target_id), None)
        if safety_active:
            if g.kind != GoalType.REST:
                self._finish("SAFETY_INTERRUPTION", target)
            return
        # A sound is a bounded orienting goal; repeat events have distinct IDs.
        if acoustic is not None and g.kind != GoalType.ORIENT:
            if g.kind != GoalType.REST:
                self.interrupted += 1
            self.goal = Goal(GoalType.ORIENT, acoustic.target_id, deadline=2.0)
            return
        if g.kind == GoalType.REST:
            motion_wakeup = g.elapsed >= .3 and any(
                t.visible and t.speed > .15 and self._available(t) for t in targets)
            if (g.elapsed < self.rest_seconds and not motion_wakeup) or scenario == "S0" or readiness < .25:
                return
            candidates = [t for t in targets if self._available(t)]
            humans = [t for t in candidates if t.kind == "human"]
            objects = [t for t in candidates if t.kind != "human"]
            target = humans[0] if humans and (social_need >= .2 or not objects) else (
                min(objects, key=lambda t: t.distance) if objects else None)
            if target:
                kind = (GoalType.SOCIAL if target.kind == "human" else
                        GoalType.TRACK if target.speed > .05 else GoalType.INSPECT)
                self.goal = Goal(kind, target.target_id, initial_distance=target.distance,
                                 previous_distance=target.distance,
                                 deadline=12.0 if kind == GoalType.SOCIAL else 20.0,
                                 look_away_sign=-1.0 if target.bearing > 0 else 1.0)
            elif self.time >= self.next_explore_time and (scenario == "S1" or curiosity >= .65):
                self.goal = Goal(GoalType.EXPLORE, deadline=6.0)
            return
        if g.kind == GoalType.EXPLORE:
            if g.elapsed >= g.deadline:
                # Completion requires unseen view signatures from feedback().
                self._finish("SUCCESS" if g.progress >= .5 else "NO_INFORMATION")
            return
        if g.kind == GoalType.ORIENT:
            target = acoustic
            if target is None:
                self._finish("SUCCESS" if g.aligned_time >= .2 else "SOUND_EXPIRED")
                return
        if target is None:
            g.lost_time += dt
            if g.lost_time >= 1.2:
                self._finish("TARGET_LOST")
            return
        if g.kind == GoalType.ORIENT and abs(target.gaze_error) < .35:
            self._finish("SUCCESS", target)
            return
        aligned = target.visible and abs(target.gaze_error) < .20
        g.lost_time = 0.0 if target.visible else g.lost_time + dt
        # Acquisition may use a last known bearing, but must confirm visual lock.
        if g.lost_time > 3.0 or g.elapsed >= g.deadline:
            self._finish("TARGET_LOST" if g.lost_time > 3.0 else "TIMEOUT", target)
            return
        g.aligned_time = g.aligned_time + dt if aligned else 0.0
        stop_dist = 1.0 if g.kind == GoalType.SOCIAL else .60
        if g.phase == GoalPhase.ORIENT and g.aligned_time >= .5:
            self._phase(GoalPhase.OBSERVE if g.kind in (GoalType.TRACK, GoalType.ORIENT)
                        or target.distance <= stop_dist + .08 else GoalPhase.APPROACH)
        elif g.phase == GoalPhase.APPROACH:
            self.progress_delta = float(np.clip(g.previous_distance - target.distance, -.03, .03))
            g.previous_distance = target.distance
            g.progress = float(np.clip((g.initial_distance - target.distance) /
                                      max(.1, g.initial_distance - stop_dist), 0, .8))
            if target.distance <= stop_dist + .08:
                self._phase(GoalPhase.OBSERVE)
        elif g.phase == GoalPhase.OBSERVE:
            if g.kind == GoalType.SOCIAL:
                if g.signal_started:
                    self._phase(GoalPhase.WAIT)
            elif g.kind == GoalType.ORIENT and g.aligned_time >= .2:
                self._finish("SUCCESS", target)
            elif g.kind == GoalType.TRACK:
                g.progress = min(1.0, g.progress + (dt / 3.0 if aligned else 0.0))
                if target.speed <= .05 and g.aligned_time >= 2.0:
                    self._finish("SUCCESS", target)
            elif target.distance <= .8 and g.aligned_time >= self.inspection_seconds:
                self._finish("SUCCESS", target)
        elif g.phase == GoalPhase.WAIT and g.phase_elapsed >= 2.0:
            self._finish("SUCCESS" if g.responded else "NO_RESPONSE", target)

    def feedback(self, *, signal_started=False, human_responded=False, view_signature=None):
        g = self.goal
        if g.kind == GoalType.SOCIAL:
            g.signal_started |= signal_started
            # Only attribute responses after this goal's actual signal.
            g.responded |= bool(g.signal_started and human_responded)
        if g.kind == GoalType.EXPLORE and view_signature is not None:
            signature = tuple(view_signature)
            key = ("view", signature)
            if key not in self.memory:
                self.memory[key] = (self.time, np.zeros(3))
                g.progress = min(1.0, g.progress + .1)

    def vector(self):
        v = np.zeros(12, dtype=np.float32)
        v[int(self.goal.kind)] = 1
        v[6 + int(self.goal.phase)] = 1
        v[10] = min(1, self.goal.elapsed / self.goal.deadline)
        v[11] = self.goal.progress
        return v

    def telemetry(self):
        g = self.goal
        return dict(type=g.kind.name, target_id=g.target_id, phase=g.phase.name,
                    elapsed=g.elapsed, progress=g.progress, completed=self.completed,
                    failed=self.failed, interrupted=self.interrupted,
                    last_result=self.last_result, completion_event=self.completion_event,
                    failure_event=self.failure_event, progress_delta=self.progress_delta)
