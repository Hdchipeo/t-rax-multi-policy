"""Bounded skills with a small learned residual; all output passes safety."""
import numpy as np
from trax_rl.dynamics.goal_manager import GoalType, GoalPhase


class GoalSkillPolicy:
    def act(self, goal, target, *, pan, tilt, camera_error=0.0):
        action = np.zeros(5, dtype=np.float32)
        action[4] = .1
        if goal.kind == GoalType.REST:
            # Hold position, not perpetual recentering during rest.
            if target is not None:
                away_pan = np.clip(-target.bearing + goal.look_away_sign * .5, -.75, .75)
                error = away_pan - pan
                action[2] = np.clip(error * 1.8, -.35, .35) if abs(error) > .04 else 0
            return action
        if goal.kind == GoalType.EXPLORE:
            # A bounded sweep; no blind locomotion needed to earn novelty.
            scan_target = .5 * np.sin(goal.elapsed * .8)
            action[2] = np.clip((scan_target - pan) * 1.8, -.4, .4)
            return action
        if target is None:
            return action
        action[2] = np.clip(-camera_error * 1.8, -.6, .6) if abs(camera_error) > .04 else 0
        action[3] = np.clip(-tilt * 1.2, -.3, .3) if abs(tilt) > .04 else 0
        # Head first; rotate body only to bring a wide bearing into reach, or approach.
        if abs(target.bearing) > .65 or goal.phase == GoalPhase.APPROACH:
            action[1] = np.clip(-target.bearing * .9, -.5, .5)
        # Recenter the body before the head reaches its mechanical boundary.
        if abs(pan) > .7:
            action[1] = np.clip(pan * .8, -.5, .5)
        if goal.phase == GoalPhase.APPROACH and target.visible:
            stop = 1.0 if goal.kind == GoalType.SOCIAL else .60
            speed = max(.035, (target.distance - stop) * .3) if target.distance > stop + .06 else 0.0
            action[0] = np.clip(speed, 0, .18) * max(0, 1 - abs(target.bearing) / .6)
        if goal.kind == GoalType.SOCIAL and goal.phase == GoalPhase.OBSERVE:
            action[4] = .5
            # One deliberate nod; the next step enters WAIT and stops signalling.
            action[3] = -.18
        if goal.kind == GoalType.SOCIAL and goal.phase == GoalPhase.WAIT:
            action[0:2] = 0
            action[3] = np.clip(-tilt * 2.0, -.3, .3) if abs(tilt) > .01 else 0
            if goal.phase_elapsed > 1.2:
                away_error = goal.look_away_sign * .35 - camera_error
                action[2] = np.clip(away_error * 1.8, -.3, .3)
        return action

    def residual(self, nominal, learned_action, goal):
        action = nominal.copy()
        if goal.kind == GoalType.REST:
            return action
        # PPO continuous output is a signed correction, including attention.
        correction = np.clip(np.asarray(learned_action, dtype=np.float32), -1, 1)
        action[:4] += correction[:4] * np.array([.04, .08, .08, .06])
        if goal.phase != GoalPhase.APPROACH:
            action[0] = 0  # observation and turn-taking must not drift forward
        if goal.kind == GoalType.SOCIAL and goal.phase == GoalPhase.OBSERVE:
            action[4] = np.clip(action[4] + correction[4] * .08, .42, .58)
        return action
