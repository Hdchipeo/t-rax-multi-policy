import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env
from trax_rl.envs.trax_desk_env import TraxDeskEnv
from trax_rl.envs.trax_desk_social_env import TraxDeskSocialEnv


class DeskSocialRLTests(unittest.TestCase):
    def setUp(self):self.env=TraxDeskSocialEnv(horizon=20)
    def tearDown(self):self.env.close()

    def fake_observation(self):
        observation=self.env.env._observation()
        observation['robot'][:2]=0
        observation['humans'][:]=0
        observation['objects'][:]=0
        return observation

    def reward(self, observation):
        info=self.env.env._info()
        info['simulation_unstable']=False
        with patch.object(self.env,'_camera_error',return_value=np.zeros(2)):
            return self.env._task_reward(observation,info,np.zeros(5))

    def test_gym_contract_and_seeded_curriculum(self):
        check_env(self.env,skip_render_check=True)
        env=TraxDeskSocialEnv(training=True,horizon=20)
        try:
            samples=[]
            for _ in range(2):
                obs,info=env.reset(seed=60)
                self.assertTrue(env.observation_space.contains(obs))
                samples.append((env.env.data.qpos.copy(),info['human_scenario']))
            np.testing.assert_array_equal(samples[0][0],samples[1][0])
            self.assertEqual(samples[0][1],samples[1][1])
        finally:env.close()

    def test_visibility_required_and_dwell_resets_when_lost(self):
        self.env.reset(seed=42)
        obs=self.fake_observation()
        for _ in range(20):self.reward(obs)
        self.assertFalse(self.env.noticed.any())
        obs['humans'][0,0]=1
        for _ in range(4):self.reward(obs)
        obs['humans'][0,0]=0
        self.reward(obs)
        obs['humans'][0,0]=1
        for _ in range(4):self.reward(obs)
        self.assertFalse(self.env.noticed[0])
        for _ in range(5):self.reward(obs)
        self.assertTrue(self.env.noticed[0])
        for _ in range(100):self.reward(obs)
        self.assertEqual(self.env.metrics['people_noticed'],1)

    def test_wave_only_after_event_and_counts_once(self):
        self.env.reset(seed=42)
        obs=self.fake_observation()
        obs['humans'][0,0]=1
        for _ in range(10):self.reward(obs)
        self.assertEqual(self.env.metrics['wave_responses'],0)
        self.env.env.humans.new_events=[dict(person_id=1,action='wave',time_s=0.)]
        for _ in range(100):self.reward(obs)
        self.assertEqual(self.env.metrics['wave_opportunities'],1)
        self.assertEqual(self.env.metrics['wave_responses'],1)

    def test_inspect_requires_release_stillness_and_counts_once(self):
        self.env.reset(seed=42)
        obs=self.fake_observation()
        obs['objects'][0]=[1,0,.2,1,0,0]
        for _ in range(20):self.reward(obs)
        self.assertEqual(self.env.metrics['objects_inspected'],0)
        self.env.env.humans.released[0]=True
        obs['objects'][0,3]=0
        obs['robot'][0]=1
        for _ in range(20):self.reward(obs)
        self.assertEqual(self.env.metrics['objects_inspected'],0)
        obs['robot'][0]=0
        for _ in range(20):self.reward(obs)
        self.assertEqual(self.env.metrics['objects_inspected'],1)
        for _ in range(100):self.reward(obs)
        self.assertEqual(self.env.metrics['objects_inspected'],1)

    def test_curriculum_late_start_keeps_props_and_limits_horizon(self):
        obs,info=self.env.reset(seed=42,options={'start_time':82})
        self.assertEqual(info['human_scenario']['objects'],dict(water_glass='on_table',new_object='on_table'))
        self.assertFalse(self.env.env.humans.present.any())
        np.testing.assert_array_equal(obs['memory'],0)
        for step in range(20):
            _,_,term,trunc,_=self.env.step(np.zeros(5))
            self.assertFalse(term)
            self.assertEqual(trunc,step==19)
        self.env.reset(seed=42)
        self.assertEqual(self.env.metrics['steps'],0)
        self.assertFalse(self.env.inspected.any())
        self.assertFalse(self.env.discovered.any())

    def test_actor_has_no_invisible_target_or_schedule_fields(self):
        self.env.reset(seed=42)
        obs=self.fake_observation()
        self.env.env.humans.heads[:]=[1.,-1.,1.6]
        augmented=self.env._augment(obs)
        np.testing.assert_array_equal(augmented['elevation'],0)
        np.testing.assert_array_equal(augmented['memory'],0)
        self.assertEqual(set(augmented),set(self.env.env.observation_space.spaces)|{'memory','elevation'})

    def test_visual_optimization_preserves_track_physics(self):
        a,b=TraxDeskEnv(animate_tracks=True),TraxDeskEnv(animate_tracks=False)
        try:
            for _ in range(20):
                a.step(np.array([.3,-.2,.1,.1,0]))
                b.step(np.array([.3,-.2,.1,.1,0]))
            np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-10)
            np.testing.assert_allclose(a.data.qvel,b.data.qvel,atol=1e-10)
        finally:a.close();b.close()


if __name__=='__main__':unittest.main()
