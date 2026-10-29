import unittest
import numpy as np
from trax_rl.dynamics.internal_drives import InternalDrivesEngine
from trax_rl.dynamics.goal_manager import GoalManager, GoalType, TargetEstimate
from trax_rl.dynamics.event_scheduler import EventScheduler
from trax_rl.dynamics.behavior_manager import BehaviorManager, Behavior
from trax_rl.dynamics.scenario_manager import ScenarioManager
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv


class DrivesTests(unittest.TestCase):
    def test_evolution_independent_of_frequency(self):
        states = []
        for dt in [.1, .05]:
            drives = InternalDrivesEngine()
            for _ in range(round(1/dt)):
                drives.step(dt, novelty_score=.6, human_visible=True,
                    human_engaged=True, action_energy=.3, is_resting=False)
            states.append(drives.get_vector())
        np.testing.assert_allclose(states[0], states[1], atol=1e-6)

    def test_rest_not_charging_presence_not_engagement(self):
        drives = InternalDrivesEngine()
        drives.reset(initial_energy=.5)
        drives.readiness = .5
        drives.step(1, 0, human_visible=True, human_engaged=False,
                    action_energy=0, is_resting=True)
        self.assertLess(drives.energy_budget, .5)
        self.assertGreater(drives.readiness, .5)
        self.assertGreaterEqual(drives.social_need, .3)


class GoalTests(unittest.TestCase):
    def target(self, distance=.6):
        return TargetEstimate("object", "object", np.array([0.,distance,0.]), distance, 0, True)

    def advance(self, manager, targets, **kwargs):
        manager.step(.1, targets, scenario="S2", curiosity=.5,
                     social_need=.3, readiness=1, **kwargs)

    def test_completion_memory_and_revisit_on_change(self):
        manager = GoalManager(rest_seconds=.2, inspection_seconds=.3)
        target = self.target()
        for _ in range(12):
            self.advance(manager, [target])
        self.assertEqual(manager.completed, 1)
        for _ in range(10):
            self.advance(manager, [target])
        self.assertEqual(manager.completed, 1)
        self.assertEqual(manager.goal.kind, GoalType.REST)
        target.position[0] += .4
        self.advance(manager, [target])
        self.assertEqual(manager.goal.kind, GoalType.INSPECT)

    def test_safety_preempts_without_success(self):
        manager = GoalManager(rest_seconds=.1)
        self.advance(manager, [self.target(1.2)])
        self.advance(manager, [self.target(1.2)], safety_active=True)
        self.assertEqual(manager.goal.kind, GoalType.REST)
        self.assertEqual(manager.completed, 0)
        self.assertEqual(manager.interrupted, 1)

    def test_response_before_signal_not_attributed(self):
        manager = GoalManager(rest_seconds=.1)
        target = TargetEstimate("human", "human", np.array([0.,1.,0.]), 1., 0, True)
        self.advance(manager, [target])
        manager.feedback(human_responded=True)
        self.assertFalse(manager.goal.responded)

    def test_visible_motion_wakes_rest_but_static_target_does_not(self):
        manager = GoalManager(rest_seconds=3)
        target = self.target(1.2)
        for _ in range(4):
            self.advance(manager, [target])
        self.assertEqual(manager.goal.kind, GoalType.REST)
        target.speed = .5
        self.advance(manager, [target])
        self.assertEqual(manager.goal.kind, GoalType.TRACK)


class AcceptanceTests(unittest.TestCase):
    def sample(self, manager, **kwargs):
        values = dict(dt=.1, action=np.zeros(5), safe_action=np.zeros(5),
            overridden=False, override_reason="NONE", pan_deg=90, tilt_deg=90,
            target_visible=True, target_bearing=0., target_distance=1.2,
            human_aligned=True, human_responded=False, tier=0, is_spam=False,
            novelty_score=0.)
        values.update(kwargs)
        return manager.update_step(**values)

    def test_no_attempt_fails(self):
        manager = ScenarioManager("S6")
        for _ in range(20):
            metrics = self.sample(manager, target_is_human=True, human_responded=True)
        self.assertEqual(metrics["attention_success_rate"], 0)
        self.assertFalse(manager.evaluate_acceptance(metrics)[0])

    def test_gesture_event_and_delayed_response_count_once(self):
        manager = ScenarioManager("S6")
        for _ in range(5):
            self.sample(manager, tier=2, target_is_human=True)
        for _ in range(5):
            metrics = self.sample(manager, tier=0, target_is_human=True, human_responded=True)
        self.assertEqual(metrics["attention_attempts"], 1)
        self.assertEqual(metrics["attention_successes"], 1)

    def test_static_target_not_tracking(self):
        manager = ScenarioManager("S3")
        for _ in range(50):
            metrics = self.sample(manager)
        self.assertEqual(metrics["tracking_handoffs"], 0)
        self.assertFalse(manager.evaluate_acceptance(metrics)[0])

    def test_inspection_requires_gaze_and_distance(self):
        manager = ScenarioManager("S2")
        for _ in range(20):
            self.sample(manager, target_distance=.6, gaze_error=.8)
        self.assertEqual(manager.object_inspections, 0)
        for _ in range(20):
            self.sample(manager, target_distance=.6, gaze_error=0)
        self.assertEqual(manager.object_inspections, 1)

    def test_missing_nonfinite_metrics_fail_closed(self):
        manager = ScenarioManager("S0")
        self.assertFalse(manager.evaluate_acceptance({})[0])
        metrics = self.sample(manager)
        metrics["motion_jerk"] = float("nan")
        self.assertFalse(manager.evaluate_acceptance(metrics)[0])


class SafetyTests(unittest.TestCase):
    def test_slew_cannot_reintroduce_forbidden_reverse(self):
        manager = BehaviorManager()
        manager.current_behavior = Behavior.EXPLORE
        manager.last_safe_lin = -.3
        action, _, _ = manager.sanitize_action(-.3,0,0,0,.1,0,0,1.5, target_is_human=False)
        self.assertEqual(action[0], 0)
        self.assertIn("BLIND_REVERSE_CLAMP", manager.get_safety_metrics()["correction_reasons"])

    def test_stop_survives_slew(self):
        manager = BehaviorManager()
        manager.current_behavior = Behavior.SAFE_STOP
        manager.last_safe_lin = .3
        manager.last_safe_pan_v = .4
        action, _, _ = manager.sanitize_action(.3,.3,.4,.1,.1,0,0,1.5)
        np.testing.assert_array_equal(action[:4], 0)

    def test_event_emitted_once(self):
        scheduler = EventScheduler()
        scheduler.scheduled_events = [dict(time=0, type="ball_roll", duration=1,
                                            triggered=False, pos=np.zeros(3))]
        self.assertIsNotNone(scheduler.update(0,.1))
        self.assertIsNone(scheduler.update(.1,.1))


class IntegrationTests(unittest.TestCase):
    def test_acoustic_direction_updates_when_body_turns(self):
        env = TraxBehavioralEnv(controller_mode="goal_residual", scenario="S4", enable_domain_randomization=False)
        try:
            env.reset(seed=42)
            env.scenario_manager.trigger_acoustic_event(np.array([1.,1.,.5]), duration=5)
            env.data.qpos[env.base_yaw_qpos_adr] = -np.pi / 4
            env.step(np.zeros(5))
            self.assertAlmostEqual(env.scenario_manager.acoustic_direction, 0, places=4)
        finally:
            env.close()

    def test_seeded_rollouts_reproducible(self):
        trajectories = []
        for _ in range(2):
            env = TraxBehavioralEnv(controller_mode="goal_residual", scenario="S7")
            try:
                obs, _ = env.reset(seed=1001)
                self.assertEqual(obs.shape, (54,))
                trajectory = []
                for _ in range(100):
                    obs, reward, term, trunc, info = env.step(np.zeros(5))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(np.isfinite(reward))
                    trajectory.append(obs.copy())
                trajectories.append(trajectory)
            finally:
                env.close()
        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_inspection_completes_then_rests(self):
        env = TraxBehavioralEnv(controller_mode="goal_residual", scenario="S2", enable_domain_randomization=False)
        try:
            env.reset(seed=42)
            completed = False
            for _ in range(200):
                _, _, _, _, info = env.step(np.zeros(5))
                if info["goal_metrics"]["completion_event"]:
                    completed = True
                    self.assertEqual(info["goal_metrics"]["type"], "REST")
                    break
            self.assertTrue(completed)
            self.assertGreaterEqual(info["scenario_metrics"]["object_inspections"], 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
