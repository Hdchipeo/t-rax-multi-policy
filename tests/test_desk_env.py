import unittest
import numpy as np
from scripts.run_desk import prepare_mujoco_runtime  # noqa: F401
import mujoco
from trax_rl.envs.trax_desk_env import TraxDeskEnv
from scripts.run_desk import DeskDemo


class DeskTests(unittest.TestCase):
    def test_scene_layout_and_scale(self):
        env=TraxDeskEnv()
        try:
            top=env.model.geom('desk_top')
            np.testing.assert_allclose(top.size[:2],[.7,.375])
            self.assertAlmostEqual(float(top.pos[2]+top.size[2]),.76)
            for name in ['monitor','keyboard','mouse','succulent','mug','desk_ornament']:
                self.assertGreater(env.model.body(name).id,0)
            self.assertEqual(env.model.nu,2)
            self.assertEqual(env.observation_space['robot'].shape,(10,))
        finally:env.close()

    def test_stationary_robot_is_supported_and_finite(self):
        env=TraxDeskEnv()
        try:
            for _ in range(300):
                obs,_,term,_,info=env.step(np.zeros(5))
                self.assertFalse(term)
                self.assertTrue(env.observation_space.contains(obs))
            self.assertGreater(info['pose']['z'],env.TABLE_HEIGHT)
            np.testing.assert_array_equal(obs['sensors'][1:],1)
            self.assertFalse(info['collisions'])
        finally:env.close()

    def test_forward_motion_uses_physics(self):
        env=TraxDeskEnv()
        try:
            before=env._pose()[0]
            for _ in range(40):env.step(np.array([.5,0,0,0,0]))
            self.assertGreater(env._pose()[0][1]-before[1],.03)
            self.assertTrue(np.all(np.isfinite(env.data.qvel)))
        finally:env.close()

    def test_differential_drive_turns_body(self):
        env=TraxDeskEnv()
        try:
            for _ in range(60):env.step(np.array([0,-.65,0,0,0]))
            self.assertLess(env._pose()[1],-.3)
        finally:env.close()

    def test_zone_guard_stops_forward_drive(self):
        env=TraxDeskEnv()
        try:
            reasons=set()
            for _ in range(400):
                _,_,term,_,info=env.step(np.array([1,0,0,0,0]))
                reasons.add(info['safety_reason'])
                self.assertFalse(term)
                margin=np.min(env.ZONE_HALF-np.abs(env._pose()[0][:2]-env.ZONE_CENTER))-env.ROBOT_RADIUS
                self.assertGreater(margin,0)
            self.assertIn('ZONE_BOUNDARY',reasons)
        finally:env.close()

    def test_no_invisible_wall_robot_can_fall_when_guard_disabled(self):
        env=TraxDeskEnv(restrict_to_zone=False,safety_enabled=False)
        try:
            fell=False
            for _ in range(400):
                _,_,term,_,info=env.step(np.array([1,0,0,0,0]))
                if term:
                    fell=info['fell'];break
            self.assertTrue(fell)
            self.assertFalse(info['simulation_unstable'])
        finally:env.close()

    def test_cliff_probes_detect_missing_table(self):
        env=TraxDeskEnv()
        try:
            env.data.qpos[env.base_adr]=.69
            mujoco.mj_forward(env.model,env.data)
            _,probes=env._sensors()
            self.assertLess(probes.min(),.5)
        finally:env.close()

    def test_real_monitor_collision_is_reported(self):
        env=TraxDeskEnv()
        try:
            env.data.qpos[env.base_adr:env.base_adr+2]=[-.20,.26]
            mujoco.mj_forward(env.model,env.data)
            self.assertIn('monitor_stem',env._contacts())
        finally:env.close()

    def test_seed_replays_ball_randomization(self):
        positions=[]
        for _ in range(2):
            env=TraxDeskEnv(domain_randomization=True)
            try:
                env.reset(seed=101)
                positions.append(env.data.xpos[env.ball_id].copy())
            finally:env.close()
        np.testing.assert_array_equal(*positions)

    def test_demo_completes_lap_without_contact_or_fall(self):
        env=TraxDeskEnv(max_episode_steps=2400)
        policy=DeskDemo()
        try:
            for _ in range(2400):
                _,_,term,_,info=env.step(policy.act(env))
                self.assertFalse(term)
                self.assertFalse(info['collisions'])
                if policy.completed_laps>0:break
            self.assertGreaterEqual(policy.completed_laps,1)
        finally:env.close()





class ContinuousTrackTests(unittest.TestCase):
    def test_two_track_contact_strips_and_no_wheel_proxy_joints(self):
        env=TraxDeskEnv()
        try:
            names=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(env.model.njnt)]
            self.assertFalse(any(name.startswith('wheel_') for name in names))
            self.assertEqual(len(env.track_geoms),2)
            for geom in env.track_geoms:
                self.assertEqual(env.model.geom_type[geom],mujoco.mjtGeom.mjGEOM_BOX)
                self.assertGreater(env.model.geom_size[geom,1],env.model.geom_size[geom,0]*3)
        finally:env.close()

    def test_equal_belts_drive_forward_and_reset_clears_motion(self):
        env=TraxDeskEnv()
        try:
            before=env._pose()[0]
            for _ in range(50):env.step(np.array([.5,0,0,0,0]))
            np.testing.assert_allclose(env.track_speeds,[.04,.04],atol=1e-8)
            np.testing.assert_allclose(env.model.geom_surfacevel[env.track_geoms,1],[-.04,-.04])
            self.assertGreater(env._pose()[0][1]-before[1],.05)
            env.reset(seed=42)
            np.testing.assert_array_equal(env.track_speeds,0)
            np.testing.assert_array_equal(env.model.geom_surfacevel[env.track_geoms],0)
        finally:env.close()

    def test_opposite_belts_turn_and_treads_animate(self):
        env=TraxDeskEnv()
        try:
            before=env.model.geom_pos[env.tread_geoms[0]].copy()
            for _ in range(60):env.step(np.array([0,-.65,0,0,0]))
            self.assertGreater(env.track_speeds[0],0)
            self.assertLess(env.track_speeds[1],0)
            self.assertLess(env._pose()[1],-.3)
            self.assertFalse(np.allclose(before,env.model.geom_pos[env.tread_geoms[0]]))
        finally:env.close()

    def test_belts_in_air_do_not_propel_the_chassis(self):
        env=TraxDeskEnv(safety_enabled=False)
        try:
            env.data.qpos[env.base_adr+2]=1.5
            mujoco.mj_forward(env.model,env.data)
            before=env._pose()[0].copy()
            for _ in range(3):env.step(np.array([1,0,0,0,0]))
            self.assertLess(abs(env._pose()[0][1]-before[1]),.001)
            self.assertLess(env._pose()[0][2],before[2])
        finally:env.close()


class CadKinematicsTests(unittest.TestCase):
    def test_cad_mesh_assets_and_pbr_materials(self):
        env = TraxDeskEnv()
        try:
            expected_meshes = [
                'trax_base_mesh', 'trax_chassis_white_mesh', 'trax_chassis_silver_mesh', 'trax_chassis_gold_mesh',
                'trax_sprocket_fl_mesh', 'trax_sprocket_fr_mesh', 'trax_sprocket_rl_mesh', 'trax_sprocket_rr_mesh',
                'trax_track_shoe_left_mesh', 'trax_track_shoe_right_mesh', 'trax_pan_bracket_mesh',
                'trax_head_mesh', 'trax_head_white_mesh', 'trax_head_silver_mesh', 'trax_head_lens_mesh',
            ]
            for mname in expected_meshes:
                mid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MESH, mname)
                self.assertGreaterEqual(mid, 0, f"Mesh {mname} not found in model")
            for mat in ['trax_black', 'trax_white', 'trax_silver', 'trax_gold', 'trax_sprocket_mat', 'trax_tread_mat', 'trax_lens_mat']:
                mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
                self.assertGreaterEqual(mat_id, 0, f"Material {mat} not found in model")
        finally:
            env.close()

    def test_exact_cad_servo_joints_and_camera_optical_center(self):
        env = TraxDeskEnv()
        try:
            pan_body = env.model.body('head_pan_link')
            np.testing.assert_allclose(pan_body.pos, [-0.000088, -0.001175, 0.021413], atol=1e-5)
            tilt_body = env.model.body('head_tilt_link')
            np.testing.assert_allclose(tilt_body.pos, [0.0, 0.0, 0.011862], atol=1e-5)

            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_camera')
            self.assertGreaterEqual(cam_id, 0)
            np.testing.assert_allclose(env.model.cam_pos[cam_id], [0.000088, 0.0265, 0.02595], atol=1e-4)

            # Check joint ranges match MG90S physical constraints (pan: +/-60 deg, tilt: +/-40 deg)
            j_pan = env.model.joint('joint_pan')
            np.testing.assert_allclose(env.model.jnt_range[j_pan.id], [-1.047, 1.047], atol=1e-2)
            j_tilt = env.model.joint('joint_tilt')
            np.testing.assert_allclose(env.model.jnt_range[j_tilt.id], [-0.698, 0.698], atol=1e-2)
        finally:
            env.close()

    def test_20_cad_track_shoes_and_differential_sprockets_animation(self):
        env = TraxDeskEnv()
        try:
            self.assertEqual(len(env.tread_geoms), 2)
            self.assertEqual(len(env.tread_geoms[0]), 20)
            self.assertEqual(len(env.tread_geoms[1]), 20)
            all_shoes = env.tread_geoms[0] + env.tread_geoms[1]
            self.assertEqual(len(all_shoes), 40)
            for gid in all_shoes:
                self.assertEqual(env.model.geom_type[gid], mujoco.mjtGeom.mjGEOM_MESH)
                self.assertEqual(env.model.geom_group[gid], 2)
                self.assertEqual(env.model.geom_contype[gid], 0)
                self.assertEqual(env.model.geom_conaffinity[gid], 0)

            # Check all 4 sprockets exist and are in visual group 2
            for sp_name in ['sprocket_front_left', 'sprocket_front_right', 'sprocket_rear_left', 'sprocket_rear_right']:
                sp_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, sp_name)
                self.assertGreaterEqual(sp_gid, 0, f"Sprocket {sp_name} missing")
                self.assertEqual(env.model.geom_group[sp_gid], 2)

            # Step turning motion and verify differential sprocket rotation
            sp_fl = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'sprocket_front_left')
            sp_fr = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'sprocket_front_right')
            q_fl_before = env.model.geom_quat[sp_fl].copy()
            q_fr_before = env.model.geom_quat[sp_fr].copy()

            for _ in range(40):
                env.step(np.array([0.0, 0.8, 0.0, 0.0, 0.0]))

            self.assertFalse(np.allclose(env.model.geom_quat[sp_fl], q_fl_before))
            self.assertFalse(np.allclose(env.model.geom_quat[sp_fr], q_fr_before))
            # Left and right sprockets rotate in opposite directions during in-place turn
            self.assertFalse(np.allclose(env.model.geom_quat[sp_fl], env.model.geom_quat[sp_fr]))

            # Reset restores track speeds and cleans up
            env.reset(seed=42)
            np.testing.assert_array_equal(env.track_speeds, 0)
        finally:
            env.close()

    def test_behavioral_and_idle_envs_with_cad_model(self):
        from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv
        from trax_rl.envs.trax_idle_env import TraxIdleEnv

        benv = TraxBehavioralEnv(controller_mode="goal_residual", scenario="S1")
        try:
            obs, info = benv.reset(seed=42)
            self.assertTrue(benv.observation_space.contains(obs))
            for _ in range(20):
                obs, reward, term, trunc, info = benv.step(np.zeros(5))
                self.assertFalse(term)
        finally:
            benv.close()

        ienv = TraxIdleEnv()
        try:
            obs, info = ienv.reset(seed=42)
            self.assertTrue(ienv.observation_space.contains(obs))
            obs, reward, term, trunc, info = ienv.step(0)
            self.assertFalse(term)
        finally:
            ienv.close()


if __name__=='__main__':unittest.main()
