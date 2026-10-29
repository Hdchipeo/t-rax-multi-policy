import unittest
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env
from trax_rl.envs.trax_desk_env import TraxDeskEnv


class DeskHumanTests(unittest.TestCase):
    def setUp(self):
        self.env=TraxDeskEnv(max_episode_steps=1900)

    def tearDown(self):
        self.env.close()

    def advance(self, seconds):
        for _ in range(round(seconds/self.env.dt)):
            obs,_,term,_,info=self.env.step(np.zeros(5))
            self.assertFalse(term)
            self.assertTrue(self.env.observation_space.contains(obs))
        return obs,info

    def test_full_workday_three_people_release_objects_and_leave(self):
        social=self.env.humans
        events=list(social.events)
        max_people=0
        seated=False
        for _ in range(1800):
            obs,_,term,_,info=self.env.step(np.zeros(5))
            self.assertFalse(term)
            self.assertFalse(info['collisions'])
            self.assertTrue(self.env.observation_space.contains(obs))
            max_people=max(max_people,int(social.present.sum()))
            seated|=social.seated[0]>.99
            events.extend(info['human_scenario']['events'])
            for k,actor in [(0,0),(1,2)]:
                if social.object_state[k]=='held':
                    pos=self.env.data.qpos[social.qadr[k]:social.qadr[k]+3]
                    np.testing.assert_allclose(social.hands[actor,1],pos+[.045,0,.015],atol=.001)
        self.assertEqual(max_people,3)
        self.assertTrue(seated)
        self.assertFalse(social.present.any())
        self.assertEqual(len(events),len(DeskTimeline())+2)
        self.assertEqual(len({(e['person_id'],e['time_s'],e['action']) for e in events}),len(events))
        for k,height in enumerate([.815,.785]):
            self.assertEqual(social.object_state[k],'on_table')
            self.assertAlmostEqual(self.env.data.xpos[social.object_ids[k],2],height,delta=.006)
            np.testing.assert_allclose(self.env.data.xpos[social.object_ids[k],:2],social.TARGETS[k,:2],atol=.005)
            geom=social.collision_geoms[k]
            self.assertTrue(any(geom in [c.geom1,c.geom2] for c in self.env.data.contact[:self.env.data.ncon]))

    def test_reset_removes_props_events_and_motion(self):
        self.advance(45)
        self.env.reset(seed=42)
        s=self.env.humans
        self.assertEqual(s.object_state,['absent','absent'])
        self.assertFalse(s.released.any())
        self.assertEqual(len(s.events),1)
        self.assertEqual(s.events[0]['action'],'arrive')
        for k,g in enumerate(s.collision_geoms):
            self.assertEqual(self.env.model.geom_contype[g],0)
            self.assertEqual(self.env.model.geom_rgba[g,3],0)
        np.testing.assert_array_equal(self.env._observation()['objects'],0)

    def test_seeded_timeline_and_randomization_replay(self):
        snapshots=[]
        self.env.domain_randomization=True
        for seed in [101,101,102]:
            self.env.reset(seed=seed)
            self.advance(32)
            snapshots.append((self.env.humans.delay,self.env.humans.heads.copy(),list(self.env.humans.events)))
        self.assertEqual(snapshots[0][0],snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1],snapshots[1][1])
        self.assertEqual(snapshots[0][2],snapshots[1][2])
        self.assertNotEqual(snapshots[0][0],snapshots[2][0])

    def test_carried_glass_matches_hand_and_release_is_continuous(self):
        self.advance(21.95)
        s=self.env.humans
        before=self.env.data.qpos[s.qadr[0]:s.qadr[0]+3].copy()
        np.testing.assert_allclose(s.hands[0,1],before+[.045,0,.015],atol=.001)
        self.assertEqual(s.object_state[0],'held')
        self.assertEqual(self.env.model.geom_contype[s.collision_geoms[0]],0)
        _,_,_,_,info=self.env.step(np.zeros(5))
        self.assertEqual(s.object_state[0],'on_table')
        self.assertLess(np.linalg.norm(self.env.data.xpos[s.object_ids[0]]-before),.01)
        self.assertIn('release_water_glass',[e['action'] for e in info['human_scenario']['events']])

    def test_perception_masks_unseen_people_and_future_objects(self):
        obs,_=self.env.reset(seed=42)
        np.testing.assert_array_equal(obs['humans'][1:],0)
        np.testing.assert_array_equal(obs['objects'],0)
        # Put a head directly along the camera optical axis, then behind it.
        s=self.env.humans
        camera=s.camera
        origin=self.env.data.cam_xpos[camera].copy()
        direction=-self.env.data.cam_xmat[camera].reshape(3,3)[:,2]
        for sign,expected in [(1,True),(-1,False)]:
            target=origin+sign*direction*.35
            s._part(0,'head',target)
            mujoco.mj_forward(self.env.model,self.env.data)
            seen,_,_=s._visible(target,s.body_ids[0])
            self.assertEqual(seen,expected)
        # A new object in front occludes the same head, despite being in the FOV.
        target=origin+direction*.4
        s._part(0,'head',target)
        s._set_object(1,origin+direction*.2,collision=True)
        mujoco.mj_forward(self.env.model,self.env.data)
        self.assertFalse(s._visible(target,s.body_ids[0])[0])

    def test_hand_safety_stops_both_tracks(self):
        s=self.env.humans
        s.hands[0,0]=self.env._pose()[0]+[0,.08,.03]
        _,_,_,_,info=self.env.step(np.array([1,1,0,0,0]))
        self.assertEqual(info['safety_reason'],'HUMAN_HAND_NEAR')
        np.testing.assert_array_equal(self.env.track_speeds,0)

    def test_disabled_timeline_baseline(self):
        env=TraxDeskEnv(human_scenarios=False)
        try:
            obs,info=env.reset(seed=42)
            self.assertIsNone(info['human_scenario'])
            np.testing.assert_array_equal(obs['humans'],0)
            np.testing.assert_array_equal(obs['objects'],0)
        finally:env.close()

    def test_gym_contract_and_seeded_step(self):
        check_env(self.env,skip_render_check=True)


def DeskTimeline():
    from trax_rl.dynamics.desk_humans import DeskHumans
    return DeskHumans.TIMELINE


if __name__=='__main__':unittest.main()
