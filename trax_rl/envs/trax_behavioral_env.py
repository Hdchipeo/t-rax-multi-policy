from typing import Tuple, Dict, Any, List, Optional
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

from trax_rl.dynamics.internal_drives import InternalDrivesEngine
from trax_rl.dynamics.curiosity_engine import EpisodicNoveltyEngine
from trax_rl.dynamics.social_dynamics import HumanAgent, SocialAttentionDynamics
from trax_rl.dynamics.event_scheduler import EventScheduler
from trax_rl.dynamics.behavior_manager import BehaviorManager, Behavior
from trax_rl.dynamics.scenario_manager import ScenarioManager, BehavioralScenario, SafeCommand, TrackingRequest, TrackingStatus
from trax_rl.rewards.behavioral_reward_mixer import BehavioralRewardMixer
from trax_rl.observations.schemas_v2 import build_behavioral_observation_dict, flatten_behavioral_observation
from trax_rl.dynamics.goal_manager import GoalManager, GoalType, TargetEstimate
from trax_rl.dynamics.track_kinematics import TraxTrackAnimator
from trax_rl.policies.goal_skill_policy import GoalSkillPolicy

class TraxBehavioralEnv(gym.Env):
    """
    T-RAX Autonomous Behavioral Dynamics Gymnasium Environment (Doc 01 - 13).
    Mô phỏng toàn diện:
      - Vật lý MuJoCo chính xác cao (Khung robot T-RAX thật, cơ cấu Pan/Tilt, Camera)
      - 5 Homeostatic Internal Drives (Curiosity, Boredom, Social Need, Energy, Confidence)
      - Episodic Novelty & Anti-Stare Decay Memory (Doc 05)
      - Human Agent & 5-tier Attention Dynamics (Doc 06)
      - Stochastic Event Scheduler (Doc 02)
      - Composite Multi-Objective Reward Mixer (Doc 04)
      - Deterministic Safety Layer & Behavior Manager Arbiter (Doc 07)
      - Scenario Manager & Lifecycle Arbiter S0-S7 (Doc 13)
      - Domain Randomization (Doc 10)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 xml_path: str = None,
                 frameskip: int = 10,
                 max_episode_steps: int = 500,
                 action_mode: str = "continuous",
                 enable_domain_randomization: bool = True,
                 scenario: str = "S7",
                 randomize_scenarios: bool = False,
                 controller_mode: str = "rl",
                 reward_components: Optional[dict] = None,
                 goal_options: Optional[dict] = None,
                 timestep: Optional[float] = None):
        super().__init__()

        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            xml_path = os.path.join(base_dir, "assets", "trax", "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        if timestep is not None:
            if timestep <= 0:
                raise ValueError("timestep must be positive")
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.frameskip = frameskip
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.enable_domain_randomization = enable_domain_randomization
        self.randomize_scenarios = randomize_scenarios
        if controller_mode not in ("rl", "goal_residual"):
            raise ValueError("controller_mode must be rl or goal_residual")
        if controller_mode == "goal_residual" and action_mode != "continuous":
            raise ValueError("goal_residual requires continuous actions")
        self.controller_mode = controller_mode
        self.goal_manager = GoalManager(**(goal_options or {}))
        self.goal_skill = GoalSkillPolicy()
        self.scenario_manager = ScenarioManager(scenario=scenario)

        # Action Space:
        # Continuous (5-dim): [linear_vel, angular_vel, pan_vel, tilt_vel, attention_intensity]
        if self.action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0,
                              -1.0 if controller_mode == "goal_residual" else 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Discrete Macro Action (5 actions compatible with v0.1)
            self.action_space = spaces.Discrete(5)

        # Observation Space: Box 39-dim [-1.0 .. 1.0] (Doc 03)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(54 if controller_mode == "goal_residual" else 39,),
            dtype=np.float32
        )

        # Joint IDs & Address Mapping
        self.pan_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_pan")
        self.tilt_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_tilt")
        self.pan_qpos_adr = self.model.jnt_qposadr[self.pan_joint_id]
        self.tilt_qpos_adr = self.model.jnt_qposadr[self.tilt_joint_id]
        self.target_mocap_id = self.model.body("target").mocapid[0]
        try:
            self.human_mocap_id = self.model.body("human").mocapid[0]
        except Exception:
            self.human_mocap_id = -1

        # Trạng thái mô phỏng quả bóng lăn 2D trên sàn (Rolling Ball Dynamics - Doc 02 & Doc 13)
        self.ball_pos = np.array([0.0, 1.2, 0.08], dtype=np.float32)
        self.ball_vel = np.array([0.55, 0.35, 0.0], dtype=np.float32)
        self.has_rolling_ball = False

        # Base locomotion joints (Planar X, Y, Yaw)
        base_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
        base_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")
        base_yaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
        self.has_mobile_base = (base_x_id != -1 and base_y_id != -1 and base_yaw_id != -1)
        if self.has_mobile_base:
            self.base_x_qpos_adr = self.model.jnt_qposadr[base_x_id]
            self.base_y_qpos_adr = self.model.jnt_qposadr[base_y_id]
            self.base_yaw_qpos_adr = self.model.jnt_qposadr[base_yaw_id]
        else:
            self.base_x_qpos_adr = -1
            self.base_y_qpos_adr = -1
            self.base_yaw_qpos_adr = -1

        # Obstacle & Novel Object Bodies
        self.obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_cube")
        self.novel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "novel_object")
        self.track_animator = TraxTrackAnimator(self.model)
        self.track_phase = np.zeros(2, dtype=np.float64)

        self.viewer = None

        # Dynamic behavioral sub-engines
        self.drives = InternalDrivesEngine()
        self.novelty_engine = EpisodicNoveltyEngine(feature_dim=6, max_memory_size=1024)
        self.human_agent = HumanAgent()
        self.social_dynamics = SocialAttentionDynamics()
        self.event_scheduler = EventScheduler(mean_interval_sec=8.0)
        self.reward_mixer = BehavioralRewardMixer(**(reward_components or {}))
        self.behavior_manager = BehaviorManager()

        # Step tracking
        self.step_count = 0
        self.sim_time = 0.0
        self.time_since_motion = 0.0
        self.time_since_new_object = 0.0
        self.time_since_base_motion = 0.0
        self.recent_actions = []

        # Domain Randomization properties
        self.motor_gain = 1.0
        self.servo_delay_steps = 0
        self.tof_noise_std = 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.has_mobile_base:
            self.data.qpos[self.base_x_qpos_adr] = 0.0
            self.data.qpos[self.base_y_qpos_adr] = 0.0
            self.data.qpos[self.base_yaw_qpos_adr] = 0.0
        self.track_phase[:] = 0.0
        self.track_animator.update(self.track_phase)

        self.step_count = 0
        self.sim_time = 0.0
        self.time_since_motion = 0.0
        self.time_since_new_object = 0.0
        self.time_since_base_motion = 0.0
        self.recent_actions.clear()
        self.goal_manager.reset()

        # Reset sub-engines
        self.drives.reset()
        self.novelty_engine.reset()
        self.social_dynamics.reset()
        self.reward_mixer.reset()
        self.behavior_manager.reset()
        self.human_agent.rng = self.np_random
        self.event_scheduler.rng = self.np_random
        self.event_scheduler.generate_episode_events(episode_length_sec=self.max_episode_steps *
                                                     self.frameskip * self.model.opt.timestep)

        # Cấu hình kịch bản S0-S7 (Doc 13 Section 3-12)
        if options and "scenario" in options:
            self.scenario_manager.scenario = self.scenario_manager._parse_scenario(options["scenario"])
        elif self.randomize_scenarios:
            all_sc = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
            sampled_sc = str(self.np_random.choice(all_sc))
            self.scenario_manager.scenario = self.scenario_manager._parse_scenario(sampled_sc)

        self.scenario_manager.reset()
        scenario_cfg = self.scenario_manager.configure_environment(self)

        # Domain Randomization (Doc 10)
        sc = self.scenario_manager.scenario
        self.has_rolling_ball = (sc in (BehavioralScenario.S3_DYNAMIC_OBJECT, BehavioralScenario.S7_MIXED))

        if self.enable_domain_randomization and sc == BehavioralScenario.S7_MIXED:
            self.motor_gain = float(self.np_random.uniform(0.90, 1.10))
            self.tof_noise_std = float(self.np_random.uniform(0.005, 0.020))
            human_present = bool(self.np_random.uniform() > 0.25)
            start_angle = float(self.np_random.uniform(-0.6, 0.6))
            start_dist = float(self.np_random.uniform(0.90, 1.45))
            hx = float(np.clip(np.sin(start_angle) * start_dist, -1.2, 1.2))
            hy = float(np.clip(np.cos(start_angle) * start_dist, 0.8, 1.45))
            human_pos = np.array([hx, hy, 0.0], dtype=np.float32)
            self.human_agent.reset(is_present=human_present, custom_pos=human_pos)
        else:
            self.motor_gain = 1.0
            self.tof_noise_std = 0.01
            # Giữ nguyên cấu hình kịch bản chuyên biệt từ ScenarioManager
            if scenario_cfg.get("has_human", False):
                default_hpos = scenario_cfg.get("target_pos", np.array([0.0, 1.2, 0.0], dtype=np.float32)).copy()
                default_hpos[2] = 0.0
                self.human_agent.reset(is_present=True, custom_pos=default_hpos)
            else:
                self.human_agent.reset(is_present=False)

        # Cấu hình vị trí và vận tốc bóng lăn 2D trên mặt đất (Z = 0.08m)
        if self.has_rolling_ball:
            init_ang = float(self.np_random.uniform(-np.pi * 0.7, np.pi * 0.7))
            init_r = float(self.np_random.uniform(0.9, 1.5))
            bx = float(np.clip(np.sin(init_ang) * init_r, -1.5, 1.5))
            by = float(np.clip(np.cos(init_ang) * init_r, 0.6, 1.6))
            self.ball_pos = np.array([bx, by, 0.08], dtype=np.float32)

            vel_ang = float(self.np_random.uniform(0, 2 * np.pi))
            spd = float(self.np_random.uniform(0.52, 0.62))
            self.ball_vel = np.array([np.cos(vel_ang) * spd, np.sin(vel_ang) * spd, 0.0], dtype=np.float32)
            self.data.mocap_pos[self.target_mocap_id] = self.ball_pos.copy()
        elif sc == BehavioralScenario.S2_OBJECT_CURIOSITY:
            self.ball_pos = np.array([0.0, 1.2, 0.08], dtype=np.float32)
            self.ball_vel = np.zeros(3, dtype=np.float32)
            self.data.mocap_pos[self.target_mocap_id] = self.ball_pos.copy()
        elif sc in (BehavioralScenario.S5_HUMAN_PRESENCE, BehavioralScenario.S6_ATTENTION_SEEKING):
            self.ball_pos = np.array([5.0, 5.0, 0.08], dtype=np.float32)
            self.ball_vel = np.zeros(3, dtype=np.float32)
            self.data.mocap_pos[self.target_mocap_id] = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.375], dtype=np.float32)
        else:
            self.ball_pos = np.array([5.0, 5.0, 0.08], dtype=np.float32)
            self.ball_vel = np.zeros(3, dtype=np.float32)
            self.data.mocap_pos[self.target_mocap_id] = self.ball_pos.copy()

        # Hiển thị avatar người trong MuJoCo trên mặt sàn (Z=0.0) nếu có
        if self.human_mocap_id != -1:
            if self.human_agent.is_present:
                self.data.mocap_pos[self.human_mocap_id] = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.0], dtype=np.float32)
            else:
                self.data.mocap_pos[self.human_mocap_id] = np.array([5.0, 5.0, 0.0], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs(action=np.zeros(5, dtype=np.float32))
        return obs, {"scenario": self.scenario_manager.scenario.value, "description": scenario_cfg.get("description", "")}

    # Ma sát lăn của bóng: 0.55 m/s * tau 2.5s ≈ 1.4m quãng đường trước khi dừng
    BALL_FRICTION_TAU = 2.5   # s
    BALL_STOP_SPEED = 0.03    # m/s

    def _update_rolling_ball(self, dt: float):
        """
        Mô phỏng vật lý 2D quả bóng lăn trên mặt đất (Z = 0.08m) và va đập nảy lại:
        - Va đập 4 bức tường đấu trường tại biên +/- 1.88m (rebound reflection)
        - Va đập khối hộp Obstacle Cube tại [-0.5, 0.9]
        - Va đập khối trụ Novel Cylinder tại [0.6, 1.1]
        - Va đập thân xe robot T-RAX
        - Ma sát lăn: bóng chậm dần và dừng (tau = BALL_FRICTION_TAU), được đá lại bởi sự kiện ball_roll
        """
        sub_steps = 5
        dt_sub = dt / sub_steps
        WALL_BOUND = 1.88

        robot_x = float(self.data.qpos[self.base_x_qpos_adr]) if self.has_mobile_base else 0.0
        robot_y = float(self.data.qpos[self.base_y_qpos_adr]) if self.has_mobile_base else 0.0
        cube_p = self.data.xpos[self.obstacle_body_id] if self.obstacle_body_id != -1 else np.array([-0.5, 0.9, 0.04])
        nov_p = self.data.xpos[self.novel_body_id] if self.novel_body_id != -1 else np.array([0.6, 1.1, 0.06])

        for _ in range(sub_steps):
            # 1. Tích phân tọa độ chuyển động trên mặt sàn
            self.ball_pos[0] += self.ball_vel[0] * dt_sub
            self.ball_pos[1] += self.ball_vel[1] * dt_sub
            self.ball_pos[2] = 0.08  # Giữ quả bóng luôn lăn sát mặt đất

            # 2. Va đập và nảy lại từ 4 bức tường (Wall Bouncing)
            # Tường Đông/Tây (X = +-1.88m)
            if self.ball_pos[0] > WALL_BOUND:
                self.ball_pos[0] = WALL_BOUND - (self.ball_pos[0] - WALL_BOUND)
                self.ball_vel[0] = -abs(self.ball_vel[0]) * 0.98
                self.ball_vel[1] += float(self.np_random.uniform(-0.04, 0.04))
            elif self.ball_pos[0] < -WALL_BOUND:
                self.ball_pos[0] = -WALL_BOUND + (-WALL_BOUND - self.ball_pos[0])
                self.ball_vel[0] = abs(self.ball_vel[0]) * 0.98
                self.ball_vel[1] += float(self.np_random.uniform(-0.04, 0.04))

            # Tường Bắc/Nam (Y = +-1.88m)
            if self.ball_pos[1] > WALL_BOUND:
                self.ball_pos[1] = WALL_BOUND - (self.ball_pos[1] - WALL_BOUND)
                self.ball_vel[1] = -abs(self.ball_vel[1]) * 0.98
                self.ball_vel[0] += float(self.np_random.uniform(-0.04, 0.04))
            elif self.ball_pos[1] < -WALL_BOUND:
                self.ball_pos[1] = -WALL_BOUND + (-WALL_BOUND - self.ball_pos[1])
                self.ball_vel[1] = abs(self.ball_vel[1]) * 0.98
                self.ball_vel[0] += float(self.np_random.uniform(-0.04, 0.04))

            # 3. Va đập khối hộp Obstacle Cube
            dc = float(np.hypot(self.ball_pos[0] - cube_p[0], self.ball_pos[1] - cube_p[1]))
            if dc < 0.18:
                nx = (self.ball_pos[0] - cube_p[0]) / max(1e-4, dc)
                ny = (self.ball_pos[1] - cube_p[1]) / max(1e-4, dc)
                v_dot_n = self.ball_vel[0] * nx + self.ball_vel[1] * ny
                if v_dot_n < 0:
                    self.ball_vel[0] -= 2.0 * v_dot_n * nx
                    self.ball_vel[1] -= 2.0 * v_dot_n * ny
                self.ball_pos[0] = cube_p[0] + nx * 0.182
                self.ball_pos[1] = cube_p[1] + ny * 0.182

            # 4. Va đập khối trụ Novel Cylinder
            dn = float(np.hypot(self.ball_pos[0] - nov_p[0], self.ball_pos[1] - nov_p[1]))
            if dn < 0.15:
                nx = (self.ball_pos[0] - nov_p[0]) / max(1e-4, dn)
                ny = (self.ball_pos[1] - nov_p[1]) / max(1e-4, dn)
                v_dot_n = self.ball_vel[0] * nx + self.ball_vel[1] * ny
                if v_dot_n < 0:
                    self.ball_vel[0] -= 2.0 * v_dot_n * nx
                    self.ball_vel[1] -= 2.0 * v_dot_n * ny
                self.ball_pos[0] = nov_p[0] + nx * 0.152
                self.ball_pos[1] = nov_p[1] + ny * 0.152

            # 5. Va đập thân xe robot T-RAX
            dr = float(np.hypot(self.ball_pos[0] - robot_x, self.ball_pos[1] - robot_y))
            if dr < 0.18:
                nx = (self.ball_pos[0] - robot_x) / max(1e-4, dr)
                ny = (self.ball_pos[1] - robot_y) / max(1e-4, dr)
                v_dot_n = self.ball_vel[0] * nx + self.ball_vel[1] * ny
                if v_dot_n < 0:
                    self.ball_vel[0] -= 2.0 * v_dot_n * nx
                    self.ball_vel[1] -= 2.0 * v_dot_n * ny
                self.ball_pos[0] = robot_x + nx * 0.182
                self.ball_pos[1] = robot_y + ny * 0.182

            # 6. Ma sát lăn (Rolling Friction - P3): bóng chậm dần và dừng sau ~4-5s, chỉ lăn lại khi có
            # sự kiện ball_roll. Trước đây tốc độ bị ép vĩnh viễn 0.55 m/s → robot (~0.1 m/s) không bao giờ
            # tiếp cận được, APPROACH kéo dài vô tận (RC4).
            decay = float(np.exp(-dt_sub / self.BALL_FRICTION_TAU))
            self.ball_vel[0] *= decay
            self.ball_vel[1] *= decay
            if float(np.hypot(self.ball_vel[0], self.ball_vel[1])) < self.BALL_STOP_SPEED:
                self.ball_vel[0] = 0.0
                self.ball_vel[1] = 0.0

        # Cập nhật vị trí quả bóng lên đối tượng MuJoCo
        self.data.mocap_pos[self.target_mocap_id] = self.ball_pos.copy()
        # Chỉ coi là "có chuyển động" khi bóng thực sự đang lăn
        if float(np.hypot(self.ball_vel[0], self.ball_vel[1])) > 0.05:
            self.time_since_motion = 0.0

    def _get_multi_sector_clearance(self, robot_x: float, robot_y: float, robot_yaw: float) -> Dict[str, float]:
        """
        Mô phỏng chùm cảm biến ToF đa hướng 360 độ (Upgrade Plan Section 9):
        Front (0 deg), Front-Left (+45 deg), Front-Right (-45 deg), Left (+90 deg), Right (-90 deg), Rear (180 deg).
        Tính toán cự ly thực tế tới 4 bức tường đấu trường (+-1.95m) và mọi chướng ngại vật (Cube, Cylinder).
        """
        sectors = {
            "front": 0.0,
            "front_left": np.pi / 4.0,
            "front_right": -np.pi / 4.0,
            "left": np.pi / 2.0,
            "right": -np.pi / 2.0,
            "rear": np.pi
        }
        ARENA_WALL = 1.95
        MAX_RANGE = 2.5
        results = {}

        # Obstacles list: [(pos_x, pos_y, collision_radius)]
        obs_targets = []
        if self.obstacle_body_id != -1:
            cp = self.data.xpos[self.obstacle_body_id]
            obs_targets.append((float(cp[0]), float(cp[1]), 0.183))
        if self.novel_body_id != -1:
            np_pos = self.data.xpos[self.novel_body_id]
            obs_targets.append((float(np_pos[0]), float(np_pos[1]), 0.130))

        for name, angle_offset in sectors.items():
            theta = robot_yaw + angle_offset
            # Unit direction vector: MuJoCo chassis forward is -sin(yaw) in X, +cos(yaw) in Y
            u_x = -float(np.sin(theta))
            u_y = float(np.cos(theta))

            min_dist = MAX_RANGE

            # 1. Arena Walls
            if u_y > 1e-4:
                d = (ARENA_WALL - robot_y) / u_y
                if 0.0 < d < min_dist:
                    min_dist = d
            elif u_y < -1e-4:
                d = (-ARENA_WALL - robot_y) / u_y
                if 0.0 < d < min_dist:
                    min_dist = d

            if u_x > 1e-4:
                d = (ARENA_WALL - robot_x) / u_x
                if 0.0 < d < min_dist:
                    min_dist = d
            elif u_x < -1e-4:
                d = (-ARENA_WALL - robot_x) / u_x
                if 0.0 < d < min_dist:
                    min_dist = d

            # 2. Cylindrical / Bounding obstacles
            for ox, oy, r_coll in obs_targets:
                dx = ox - robot_x
                dy = oy - robot_y
                t = dx * u_x + dy * u_y
                if t > 0.0:
                    perp_sq = (dx**2 + dy**2) - t**2
                    if perp_sq < r_coll**2:
                        hit_dist = t - np.sqrt(max(0.0, r_coll**2 - perp_sq))
                        if 0.0 < hit_dist < min_dist:
                            min_dist = hit_dist

            clearance = max(0.0, min_dist - 0.075)
            results[name] = float(np.clip(clearance, 0.0, MAX_RANGE))

        return results

    def _get_obstacle_telemetry(self, robot_x: float, robot_y: float, robot_yaw: float) -> Tuple[float, float, float]:
        """
        Tính toán khoảng cách ToF và góc lệch tới chướng ngại vật gần nhất:
        - Cube Obstacle ([-0.5, 0.9, 0.04])
        - Novel Cylinder ([0.6, 1.1, 0.06])
        - 4 Bức tường bao quanh phòng (Arena Walls tại X, Y = +-2.0m)
        Trả về: (min_clearance, lateral_offset_x, min_wall_boundary)
        """
        multi_clearance = self._get_multi_sector_clearance(robot_x, robot_y, robot_yaw)
        min_clearance = multi_clearance["front"]
        min_offset_x = 0.0

        # Lateral offset checking for front steering avoidance
        if self.obstacle_body_id != -1:
            cube_pos = self.data.xpos[self.obstacle_body_id]
            dc_x = cube_pos[0] - robot_x
            dc_y = cube_pos[1] - robot_y
            dc_rob_x = dc_x * np.cos(robot_yaw) + dc_y * np.sin(robot_yaw)
            dc_rob_y = -dc_x * np.sin(robot_yaw) + dc_y * np.cos(robot_yaw)
            if dc_rob_y > 0.02 and abs(dc_rob_x) < 0.35:
                min_offset_x = dc_rob_x

        if self.novel_body_id != -1:
            nov_pos = self.data.xpos[self.novel_body_id]
            dn_x = nov_pos[0] - robot_x
            dn_y = nov_pos[1] - robot_y
            dn_rob_x = dn_x * np.cos(robot_yaw) + dn_y * np.sin(robot_yaw)
            dn_rob_y = -dn_x * np.sin(robot_yaw) + dn_y * np.cos(robot_yaw)
            if dn_rob_y > 0.02 and abs(dn_rob_x) < 0.35:
                if abs(dn_rob_x) < abs(min_offset_x) or min_offset_x == 0.0:
                    min_offset_x = dn_rob_x

        ARENA_WALL = 1.95
        min_wall_boundary = min(ARENA_WALL - robot_x, robot_x - (-ARENA_WALL),
                                ARENA_WALL - robot_y, robot_y - (-ARENA_WALL)) - 0.075

        return float(min_clearance), float(min_offset_x), float(min_wall_boundary)

    def _target_estimates(self):
        """Synthetic perception adapter; goal/skill code never reads MuJoCo."""
        x, y, yaw = self.data.qpos[:3] if self.has_mobile_base else (0, 0, 0)
        pan = float(self.data.qpos[self.pan_qpos_adr])
        candidates = []
        def add(target_id, kind, position, speed=0.0):
            dx, dy = position[0] - x, position[1] - y
            bearing = float(np.arctan2(dx * np.cos(yaw) + dy * np.sin(yaw),
                                      -dx * np.sin(yaw) + dy * np.cos(yaw)))
            distance = float(np.hypot(dx, dy))
            error = float((bearing + pan + np.pi) % (2 * np.pi) - np.pi)
            candidates.append(TargetEstimate(target_id, kind, np.asarray(position).copy(),
                distance, bearing, distance < 3.5 and abs(error) < np.pi / 3,
                speed=speed, gaze_error=error))
        if self.human_agent.is_present:
            add("human_01", "human", np.array([*self.human_agent.position[:2], .375]))
        if self.has_rolling_ball:
            add("ball_01", "ball", self.ball_pos, float(np.linalg.norm(self.ball_vel[:2])))
        elif self.scenario_manager.scenario.value == "S2":
            add("static_object_01", "object", self.data.mocap_pos[self.target_mocap_id])
        if self.scenario_manager.acoustic_active:
            direction = self.scenario_manager.acoustic_direction
            error = float(direction + pan)
            candidates.append(TargetEstimate(
                f"sound_{self.scenario_manager.acoustic_event_count}", "sound", np.zeros(3),
                1.0, direction, True, gaze_error=error))
        return candidates

    def _goal_target(self):
        return next((t for t in self._target_estimates()
                     if t.target_id == self.goal_manager.goal.target_id), None)

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        if self.has_mobile_base:
            robot_x = float(self.data.qpos[self.base_x_qpos_adr])
            robot_y = float(self.data.qpos[self.base_y_qpos_adr])
            robot_yaw = float(self.data.qpos[self.base_yaw_qpos_adr])
            pan_rad = float(self.data.qpos[self.pan_qpos_adr])
            tilt_rad = float(self.data.qpos[self.tilt_qpos_adr])
        else:
            robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
            pan_rad = float(self.data.qpos[self.pan_joint_id])
            tilt_rad = float(self.data.qpos[self.tilt_joint_id])

        max_pan = 1.047   # ~60 deg
        max_tilt = 0.698  # ~40 deg

        # 1. Đo đạc tương đối với mục tiêu chính theo Dynamic Target Prioritization
        has_both_targets = (self.human_agent.is_present and self.has_rolling_ball)
        if has_both_targets:
            if self.drives.social_need < 0.20:
                target_pos = self.ball_pos
                is_human = False
            else:
                target_pos = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.375], dtype=np.float32)
                is_human = True
        elif self.human_agent.is_present:
            target_pos = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.375], dtype=np.float32)
            is_human = True
        elif self.has_rolling_ball:
            target_pos = self.ball_pos
            is_human = False
        else:
            target_pos = self.data.mocap_pos[self.target_mocap_id]
            is_human = False
        goal_target = self._goal_target() if self.controller_mode == "goal_residual" else None
        if goal_target is not None and goal_target.kind != "sound":
            target_pos = goal_target.position
            is_human = goal_target.kind == "human"

        dx_w = target_pos[0] - robot_x
        dy_w = target_pos[1] - robot_y
        dx_rob = dx_w * np.cos(robot_yaw) + dy_w * np.sin(robot_yaw)
        dy_rob = -dx_w * np.sin(robot_yaw) + dy_w * np.cos(robot_yaw)
        distance = float(np.sqrt(dx_rob**2 + dy_rob**2))
        angle_to_target = float(np.arctan2(dx_rob, dy_rob))
        relative_angle = angle_to_target + pan_rad

        sc_name = self.scenario_manager.scenario.value
        has_active_target = self.human_agent.is_present or self.scenario_manager.acoustic_active or (sc_name in ("S2", "S3", "S4", "S5", "S6", "S7"))
        target_visible = has_active_target and (distance < 3.5) and (abs(relative_angle) < (np.pi / 3.0))

        target_x_norm = float(np.clip(relative_angle / (np.pi / 6.0), -1.0, 1.0)) if target_visible else 0.0
        target_y_norm = float(np.clip((target_pos[2] - 0.15) / 0.30 - tilt_rad, -1.0, 1.0)) if target_visible else 0.0
        moving = self.has_rolling_ball and float(np.linalg.norm(self.ball_vel[:2])) > .05
        motion_score = 0.8 if (moving and not is_human and target_visible) else (0.2 if target_visible else 0.0)

        # 2. Cảm biến khoảng cách ToF & Chướng ngại vật đa nguồn (Cube, Cylinder, Walls)
        front_obstacle_clean, _, _ = self._get_obstacle_telemetry(robot_x, robot_y, robot_yaw)

        # 3. Vật thể lạ (Novel Object / Semantic Cylinder)
        novel_visible = False
        dist_novel = 2.5
        rel_angle_novel = 0.0
        if self.novel_body_id != -1:
            nov_pos = self.data.xpos[self.novel_body_id]
            dn_x = nov_pos[0] - robot_x
            dn_y = nov_pos[1] - robot_y
            dn_rob_x = dn_x * np.cos(robot_yaw) + dn_y * np.sin(robot_yaw)
            dn_rob_y = -dn_x * np.sin(robot_yaw) + dn_y * np.cos(robot_yaw)
            dist_novel = float(np.sqrt(dn_rob_x**2 + dn_rob_y**2))
            angle_novel = float(np.arctan2(dn_rob_x, dn_rob_y))
            rel_angle_novel = angle_novel + pan_rad
            novel_visible = (dist_novel < 2.5) and (abs(rel_angle_novel) < (np.pi / 3.5))

        # Cảm biến ToF tổng hợp: Target (Ball/Human) và mọi chướng ngại vật (Cube, Cylinder, Walls)
        front_distance_clean = min(distance if (target_visible and abs(target_x_norm) < 0.3) else 2.5, front_obstacle_clean)
        front_distance_noisy = float(np.clip(front_distance_clean + self.np_random.normal(0, self.tof_noise_std), 0.05, 3.0))
        front_distance_norm = front_distance_noisy / 3.0

        # Semantic embedding & nhận dạng thị giác cho vật thể
        if novel_visible and (not target_visible or dist_novel < distance):
            object_embedding = np.array([0.85, -0.65, 0.70, 0.45], dtype=np.float32)
            has_obj = True
            obj_x = float(np.clip(rel_angle_novel / (np.pi / 6.0), -1.0, 1.0))
            obj_conf = 0.95
        elif target_visible:
            object_embedding = np.array([0.10, -0.20, 0.50, 0.00], dtype=np.float32)
            has_obj = True
            obj_x = target_x_norm
            obj_conf = 0.88
        else:
            object_embedding = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            has_obj = False
            obj_x = 0.0
            obj_conf = 0.0

        # Compact state & Novelty evaluation
        compact_state = self.novelty_engine.compute_compact_state(
            robot_pan_norm=pan_rad / max_pan,
            robot_tilt_norm=tilt_rad / max_tilt,
            target_visible=target_visible or novel_visible,
            target_bearing_norm=obj_x,
            target_distance_norm=front_distance_norm,
            motion_score=motion_score
        )
        novelty_score, current_view_novelty, object_novelty = self.novelty_engine.evaluate(compact_state, dt=0.1)

        # Action diversity metric
        action_diversity = 0.5
        if len(self.recent_actions) >= 10:
            act_matrix = np.array(self.recent_actions[-10:])
            action_diversity = float(np.clip(np.std(act_matrix), 0.0, 1.0))

        obs_dict = build_behavioral_observation_dict(
            # Robot (8)
            head_pan_norm=pan_rad / max_pan,
            head_tilt_norm=tilt_rad / max_tilt,
            linear_velocity_norm=float(action[0]) if len(action) > 0 else 0.0,
            angular_velocity_norm=float(action[1]) if len(action) > 1 else 0.0,
            left_motor_velocity_norm=float(action[0]) - float(action[1]) * 0.5 if len(action) > 1 else 0.0,
            right_motor_velocity_norm=float(action[0]) + float(action[1]) * 0.5 if len(action) > 1 else 0.0,
            front_distance_norm=front_distance_norm,
            battery_norm=self.drives.energy_budget,
            # Vision (16)
            face_visible=target_visible and is_human,
            face_x=target_x_norm if is_human else 0.0,
            face_y=target_y_norm if is_human else 0.0,
            face_size=float(np.clip(1.0 / max(0.5, distance), 0.0, 1.0)) if (target_visible and is_human) else 0.0,
            face_confidence=0.92 if (target_visible and is_human) else 0.0,
            object_visible=has_obj and (not is_human or not has_both_targets),
            object_x=obj_x,
            object_y=target_y_norm if target_visible else 0.0,
            object_size=0.35 if has_obj else 0.0,
            object_confidence=obj_conf,
            motion_score=motion_score,
            scene_change_score=current_view_novelty,
            object_embedding=object_embedding,
            # Temporal (6)
            time_since_new_object=self.time_since_new_object,
            time_since_human_seen=self.social_dynamics.time_since_human_seen,
            time_since_human_response=self.social_dynamics.time_since_human_response,
            time_since_motion=self.time_since_motion,
            time_since_attention_signal=self.social_dynamics.time_since_audio_signal,
            time_since_base_motion=self.time_since_base_motion,
            # Internal (5)
            curiosity=self.drives.curiosity,
            boredom=self.drives.boredom,
            social_need=self.drives.social_need,
            energy_budget=self.drives.energy_budget,
            perception_confidence=self.drives.perception_confidence,
            # Exploration (4)
            current_view_novelty=current_view_novelty,
            local_region_novelty=novelty_score,
            object_novelty=object_novelty,
            recent_action_diversity=action_diversity
        )

        base = flatten_behavioral_observation(obs_dict)
        if self.controller_mode == "goal_residual":
            return np.concatenate([base, self.goal_manager.vector(), np.array([
                self.drives.fear, self.drives.arousal, self.drives.readiness], dtype=np.float32)])
        return base

    def step(self, action):
        self.step_count += 1
        dt = float(self.model.opt.timestep * self.frameskip)
        self.sim_time += dt

        # Map discrete action to continuous vector if needed
        if self.action_mode == "discrete":
            act_idx = int(action)
            # [lin, ang, pan_vel, tilt_vel, attn]
            if act_idx == 0:   # REST
                act_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif act_idx == 1: # LOOK_LEFT
                act_vec = np.array([0.0, 0.0, 0.8, 0.0, 0.3], dtype=np.float32)
            elif act_idx == 2: # LOOK_RIGHT
                act_vec = np.array([0.0, 0.0, -0.8, 0.0, 0.3], dtype=np.float32)
            elif act_idx == 3: # LOOK_CENTER
                act_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
            elif act_idx == 4: # TRACK_TARGET
                pan_rad = float(self.data.qpos[self.pan_qpos_adr if self.has_mobile_base else self.pan_joint_id])
                robot_x = float(self.data.qpos[self.base_x_qpos_adr]) if self.has_mobile_base else 0.0
                robot_y = float(self.data.qpos[self.base_y_qpos_adr]) if self.has_mobile_base else 0.0
                robot_yaw = float(self.data.qpos[self.base_yaw_qpos_adr]) if self.has_mobile_base else 0.0
                target_pos = self.data.mocap_pos[self.target_mocap_id]
                dx_w = target_pos[0] - robot_x
                dy_w = target_pos[1] - robot_y
                dx_rob = dx_w * np.cos(robot_yaw) + dy_w * np.sin(robot_yaw)
                dy_rob = -dx_w * np.sin(robot_yaw) + dy_w * np.cos(robot_yaw)
                target_pan = -float(np.arctan2(dx_rob, dy_rob))
                err = target_pan - pan_rad
                act_vec = np.array([0.0, 0.0, float(np.clip(err * 3.0, -1.0, 1.0)), 0.0, 0.5], dtype=np.float32)
            else:
                act_vec = np.zeros(5, dtype=np.float32)
        else:
            act_vec = np.asarray(action, dtype=np.float32)
        policy_action = act_vec.copy()
        if not np.all(np.isfinite(act_vec)):
            raise ValueError("Action contains NaN or Inf")

        self.recent_actions.append(act_vec.copy())
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)

        # 1. Quản lý sự kiện môi trường thời gian thực (Event Scheduler)
        event = self.event_scheduler.update(self.sim_time, dt)
        if event is not None:
            if event["type"] == "human_move":
                h_pos = event["pos"].copy()
                h_pos[2] = 0.0
                self.human_agent.position = h_pos
                if self.human_mocap_id != -1:
                    self.data.mocap_pos[self.human_mocap_id] = h_pos
                self.time_since_motion = 0.0
            elif event["type"] == "ball_roll":
                if self.has_rolling_ball:
                    rand_ang = float(self.np_random.uniform(0, 2 * np.pi))
                    spd = float(self.np_random.uniform(0.55, 0.70))
                    self.ball_vel = np.array([np.cos(rand_ang) * spd, np.sin(rand_ang) * spd, 0.0], dtype=np.float32)
                self.time_since_motion = 0.0
                self.time_since_new_object = 0.0
            elif event["type"] == "sound_cue":
                self.scenario_manager.trigger_acoustic_event(event["pos"], duration=event.get("duration", 1.5))

        # Cập nhật vật lý 2D quả bóng lăn trên mặt đất và va đập nảy lại
        if self.has_rolling_ball:
            self._update_rolling_ball(dt)

        # 2. Lớp kiểm định an toàn (Deterministic Safety Layer)
        current_pan = float(self.data.qpos[self.pan_qpos_adr if self.has_mobile_base else self.pan_joint_id])
        current_tilt = float(self.data.qpos[self.tilt_qpos_adr if self.has_mobile_base else self.tilt_joint_id])

        # Đo khoảng cách thực tế đa nguồn tới chướng ngại vật & tường bao (Cube, Cylinder, Walls)
        robot_x = float(self.data.qpos[self.base_x_qpos_adr]) if self.has_mobile_base else 0.0
        robot_y = float(self.data.qpos[self.base_y_qpos_adr]) if self.has_mobile_base else 0.0
        robot_yaw = float(self.data.qpos[self.base_yaw_qpos_adr]) if self.has_mobile_base else 0.0

        front_obstacle_m, front_obstacle_x, min_wall_boundary = self._get_obstacle_telemetry(robot_x, robot_y, robot_yaw)
        if self.scenario_manager.acoustic_active and self.scenario_manager.acoustic_position is not None:
            sound = self.scenario_manager.acoustic_position
            dx_s, dy_s = sound[0] - robot_x, sound[1] - robot_y
            self.scenario_manager.acoustic_direction = float(np.arctan2(
                dx_s * np.cos(robot_yaw) + dy_s * np.sin(robot_yaw),
                -dx_s * np.sin(robot_yaw) + dy_s * np.cos(robot_yaw)))

        if self.controller_mode == "goal_residual":
            estimates = self._target_estimates()
            acoustic = next((t for t in estimates if t.kind == "sound"), None)
            self.goal_manager.step(dt, [t for t in estimates if t.kind != "sound"],
                scenario=self.scenario_manager.scenario.value,
                curiosity=self.drives.curiosity, social_need=self.drives.social_need,
                readiness=self.drives.readiness,
                safety_active=front_obstacle_m < .22 or self.drives.fear >= .7,
                acoustic=acoustic)

        # 1. Đo đạc tương đối với bóng lăn (Ball Telemetry)
        if self.has_rolling_ball:
            ball_p = self.ball_pos
            dx_b = ball_p[0] - robot_x
            dy_b = ball_p[1] - robot_y
            dx_b_rob = dx_b * np.cos(robot_yaw) + dy_b * np.sin(robot_yaw)
            dy_b_rob = -dx_b * np.sin(robot_yaw) + dy_b * np.cos(robot_yaw)
            ball_dist = float(np.sqrt(dx_b_rob**2 + dy_b_rob**2))
            ball_pan = -float(np.arctan2(dx_b_rob, dy_b_rob))
            ball_bearing = float(np.arctan2(dx_b_rob, dy_b_rob))
        else:
            ball_dist = 5.0
            ball_pan = 0.0
            ball_bearing = 0.0

        # 2. Đo đạc tương đối với con người (Human Telemetry)
        if self.human_agent.is_present:
            dx_h = self.human_agent.position[0] - robot_x
            dy_h = self.human_agent.position[1] - robot_y
            dx_h_rob = dx_h * np.cos(robot_yaw) + dy_h * np.sin(robot_yaw)
            dy_h_rob = -dx_h * np.sin(robot_yaw) + dy_h * np.cos(robot_yaw)
            human_dist = float(np.sqrt(dx_h_rob**2 + dy_h_rob**2))
            h_pan = -float(np.arctan2(dx_h_rob, dy_h_rob))
            h_bearing = float(np.arctan2(dx_h_rob, dy_h_rob))
            human_aligned = bool(abs(h_pan - current_pan) < 0.25)
        else:
            human_dist = 5.0
            h_pan = 0.0
            h_bearing = 0.0
            human_aligned = False

        # 3. Dynamic Target Prioritization (Ưu tiên mục tiêu có trọng số):
        # Khi cả người và bóng cùng xuất hiện (S7 Mixed):
        # - social_need < 0.20: Robot đã thỏa mãn tương tác xã hội -> ưu tiên bóng để giải tỏa buồn chán (boredom relief)
        # - social_need >= 0.20: Robot duy trì gắn kết với người (social interaction)
        has_both_targets = (self.human_agent.is_present and self.has_rolling_ball)
        if has_both_targets:
            if self.drives.social_need < 0.20:
                target_is_human = False
                target_dist = ball_dist
                target_pan = ball_pan
                target_bearing = ball_bearing
                target_pos = self.ball_pos
            else:
                target_is_human = True
                target_dist = human_dist
                target_pan = h_pan
                target_bearing = h_bearing
                target_pos = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.375], dtype=np.float32)
        elif self.human_agent.is_present:
            target_is_human = True
            target_dist = human_dist
            target_pan = h_pan
            target_bearing = h_bearing
            target_pos = np.array([self.human_agent.position[0], self.human_agent.position[1], 0.375], dtype=np.float32)
        elif self.has_rolling_ball:
            target_is_human = False
            target_dist = ball_dist
            target_pan = ball_pan
            target_bearing = ball_bearing
            target_pos = self.ball_pos
        else:
            # Mục tiêu mocap còn lại: người ảo trong S5/S6 (mocap gắn theo đầu người) hoặc đồ vật tĩnh
            target_is_human = self.scenario_manager.scenario in (BehavioralScenario.S5_HUMAN_PRESENCE, BehavioralScenario.S6_ATTENTION_SEEKING)
            target_pos = self.data.mocap_pos[self.target_mocap_id]
            dx_t = target_pos[0] - robot_x
            dy_t = target_pos[1] - robot_y
            dx_t_rob = dx_t * np.cos(robot_yaw) + dy_t * np.sin(robot_yaw)
            dy_t_rob = -dx_t * np.sin(robot_yaw) + dy_t * np.cos(robot_yaw)
            target_dist = float(np.sqrt(dx_t_rob**2 + dy_t_rob**2))
            target_pan = -float(np.arctan2(dx_t_rob, dy_t_rob))
            target_bearing = float(np.arctan2(dx_t_rob, dy_t_rob))

        if not self.human_agent.is_present:
            human_aligned = bool(abs(target_pan - current_pan) < 0.25)
        goal_target = self._goal_target() if self.controller_mode == "goal_residual" else None
        if goal_target is not None:
            target_is_human = goal_target.kind == "human"
            target_dist = goal_target.distance
            target_bearing = goal_target.bearing
            target_pan = -target_bearing
            target_pos = goal_target.position
            human_aligned = goal_target.visible and abs(goal_target.gaze_error) < .25

        # Trọng tài hành vi thời gian thực (Behavior Manager Arbiter - Upgrade Plan Section 4 & 7)
        sc_name = self.scenario_manager.scenario.value
        has_active_target = self.human_agent.is_present or self.scenario_manager.acoustic_active or (sc_name in ("S2", "S3", "S4", "S5", "S6", "S7"))

        # Ước tính mức độ nguy cơ / đe dọa (Threat Estimator - Upgrade Plan Section 3.1 & 7.4)
        threat_prox = max(0.0, (0.35 - front_obstacle_m) / 0.35) if front_obstacle_m < 0.35 else 0.0
        # Âm thanh là kích thích định hướng (orienting), không phải mối đe dọa (P5):
        # threat 0.15 * gain 3.0 / decay 0.8 → fear cân bằng ≈ 0.56 < ngưỡng ESCAPE 0.70.
        # Trước đây 0.35 → cân bằng 1.31, mọi cue > ~1s đều kích hoạt CRITICAL_THREAT (RC5).
        threat_acoustic = 0.30 if self.scenario_manager.acoustic_active else 0.0
        threat_score = float(np.clip(threat_prox * 0.7 + threat_acoustic * 0.5, 0.0, 1.0))

        ball_conf = 0.88 if (self.has_rolling_ball and ball_dist < 3.0) else (0.88 if (target_dist < 3.0 and not self.human_agent.is_present and sc_name in ("S2", "S3", "S7")) else 0.0)

        needs_safety = (front_obstacle_m < .22 or self.drives.fear >= .7 or
            self.behavior_manager.current_behavior == Behavior.RECOVERY and front_obstacle_m < .35)
        if self.controller_mode == "rl" or needs_safety:
            self.behavior_manager.arbitrate(
                dt=dt,
                emergency_stop=False,
                user_command_active=False,
                obstacle_dist_m=front_obstacle_m,
                human_engagement=self.human_agent.engagement_level if self.human_agent.is_present else 0.0,
                ball_confidence=ball_conf,
                target_visible=has_active_target and (target_dist < 3.0),
                curiosity=self.drives.curiosity,
                boredom=self.drives.boredom,
                fear=self.drives.fear,
                threat_active=bool(threat_score > 0.40),
                target_dist=target_dist,
                social_need=self.drives.social_need
            )
        else:
            self.behavior_manager.last_transition_switched = False
            self.behavior_manager.active_time_sec += dt
            # Clear a finished reflex before selecting the next goal skill.
            if self.behavior_manager.current_behavior in (Behavior.ESCAPE, Behavior.RECOVERY):
                self.behavior_manager._transition_to(Behavior.IDLE, "SAFETY_CLEARED")
        if self.controller_mode == "goal_residual":
            if self.behavior_manager.current_behavior not in (Behavior.ESCAPE, Behavior.RECOVERY):
                from trax_rl.dynamics.goal_manager import GoalPhase
                g = self.goal_manager.goal
                behavior = {
                    GoalType.REST: Behavior.IDLE, GoalType.EXPLORE: Behavior.LOOK_AROUND,
                    GoalType.ORIENT: Behavior.LOOK_AROUND, GoalType.TRACK: Behavior.TRACK,
                    GoalType.SOCIAL: Behavior.SOCIAL, GoalType.INSPECT: Behavior.BALL,
                }[g.kind]
                if g.phase == GoalPhase.APPROACH:
                    behavior = Behavior.APPROACH
                self.behavior_manager._transition_to(behavior, "GOAL_" + g.kind.name,
                                                     "goal_skill_" + g.phase.name.lower())
            nominal = self.goal_skill.act(self.goal_manager.goal, goal_target,
                pan=current_pan, tilt=current_tilt,
                camera_error=goal_target.gaze_error if goal_target else 0.0)
            act_vec = self.goal_skill.residual(nominal, policy_action, self.goal_manager.goal)

        # Reflex Expert Execution (P4): ESCAPE/RECOVERY là phản xạ an toàn tất định, không để RL quyết định.
        # Chuyên gia heuristic điều khiển đế bánh xích (lin/ang); RL vẫn giữ đầu (pan/tilt/attention)
        # để tiếp tục theo dõi kích thích. Trước đây nhãn ESCAPE không tác động đến motor (RC3).
        cur_beh = self.behavior_manager.current_behavior
        if cur_beh in (Behavior.ESCAPE, Behavior.RECOVERY):
            reflex = self.behavior_manager.policy_registry.get(cur_beh)
            if reflex is not None:
                is_fresh_escape = self.behavior_manager.last_transition_switched and (
                    getattr(self.behavior_manager, "last_behavior", None) not in (Behavior.ESCAPE, Behavior.RECOVERY)
                )
                if is_fresh_escape:
                    steer_init = -1.0 if front_obstacle_x <= 0.0 else 1.0
                    reflex.reset(turn_dir=steer_init)
                reflex_act = reflex.act(None, {
                    "front_tof_m": front_obstacle_m,
                    "front_obstacle_x": front_obstacle_x
                })
                act_vec = act_vec.copy()
                act_vec[0] = float(reflex_act[0])
                act_vec[1] = float(reflex_act[1])

        safe_action, overridden, override_reason = self.behavior_manager.sanitize_action(
            linear_vel=act_vec[0],
            angular_vel=act_vec[1],
            pan_vel=act_vec[2],
            tilt_vel=act_vec[3],
            attention_intensity=act_vec[4],
            current_pan_rad=current_pan,
            current_tilt_rad=current_tilt,
            front_obstacle_m=front_obstacle_m,
            front_obstacle_x=front_obstacle_x,
            target_dist=target_dist,
            gaze_duration=self.scenario_manager.human_gaze_duration,
            acoustic_active=self.scenario_manager.acoustic_active,
            acoustic_direction=self.scenario_manager.acoustic_direction,
            dt=dt,
            boredom=0.0 if self.scenario_manager.scenario == BehavioralScenario.S0_MOTOR_SANITY else self.drives.boredom,
            target_is_human=target_is_human,
            enable_behavior_reflexes=self.controller_mode == "rl"
        )

        # 3. Giải mã Attention và phản hồi từ HumanAgent
        tier, has_gesture, emit_audio, is_spam = self.social_dynamics.decode_attention_intensity(safe_action[4], dt)

        pan_rad = current_pan

        human_responded = self.human_agent.update(
            dt=dt,
            robot_looking_at_human=human_aligned,
            robot_head_gesture=has_gesture,
            audio_signal_emitted=emit_audio,
            signal_spam=is_spam
        )
        if self.controller_mode == "goal_residual":
            self.goal_manager.feedback(signal_started=has_gesture or emit_audio,
                human_responded=human_responded,
                view_signature=(int(current_pan / .2), int(current_tilt / .2)))

        ready_for_handoff = self.social_dynamics.step(
            dt=dt,
            human_visible=self.human_agent.is_present,
            human_aligned=human_aligned,
            human_responded=human_responded,
            engagement_level=self.human_agent.engagement_level
        )

        # 4. Điều khiển vật lý MuJoCo: Tích hợp vận tốc góc servo Pan/Tilt & Base Locomotion
        if self.has_mobile_base:
            cur_yaw = float(self.data.qpos[self.base_yaw_qpos_adr])
            dx = -float(np.sin(cur_yaw)) * float(safe_action[0]) * dt * self.motor_gain
            dy = float(np.cos(cur_yaw)) * float(safe_action[0]) * dt * self.motor_gain
            dyaw = float(safe_action[1]) * dt * self.motor_gain

            new_x = float(np.clip(self.data.qpos[self.base_x_qpos_adr] + dx, -1.90, 1.90))
            new_y = float(np.clip(self.data.qpos[self.base_y_qpos_adr] + dy, -1.90, 1.90))

            # Rào cản vật lý Kinematic 1: Khối hộp Obstacle Cube
            if self.obstacle_body_id != -1:
                cube_pos = self.data.xpos[self.obstacle_body_id]
                dist_to_cube = float(np.hypot(new_x - cube_pos[0], new_y - cube_pos[1]))
                min_safe_dist = 0.185  # chassis half-diagonal 0.070 + cube half-diagonal 0.113 + 0.002 margin
                if dist_to_cube < min_safe_dist:
                    nx = (new_x - cube_pos[0]) / dist_to_cube if dist_to_cube > 1e-4 else 0.0
                    ny = (new_y - cube_pos[1]) / dist_to_cube if dist_to_cube > 1e-4 else -1.0
                    new_x = float(cube_pos[0] + nx * min_safe_dist)
                    new_y = float(cube_pos[1] + ny * min_safe_dist)

            # Rào cản vật lý Kinematic 2: Khối trụ Novel Cylinder
            if self.novel_body_id != -1:
                nov_pos = self.data.xpos[self.novel_body_id]
                dist_to_novel = float(np.hypot(new_x - nov_pos[0], new_y - nov_pos[1]))
                min_safe_novel = 0.135  # chassis half-diagonal 0.070 + cylinder radius 0.060 + 0.005 margin
                if dist_to_novel < min_safe_novel:
                    nx = (new_x - nov_pos[0]) / dist_to_novel if dist_to_novel > 1e-4 else 0.0
                    ny = (new_y - nov_pos[1]) / dist_to_novel if dist_to_novel > 1e-4 else -1.0
                    new_x = float(nov_pos[0] + nx * min_safe_novel)
                    new_y = float(nov_pos[1] + ny * min_safe_novel)

            # Đảm bảo tuyệt đối không vượt qua biên tường 1.90m sau phản hồi
            new_x = float(np.clip(new_x, -1.90, 1.90))
            new_y = float(np.clip(new_y, -1.90, 1.90))

            prev_x = float(self.data.qpos[self.base_x_qpos_adr])
            prev_y = float(self.data.qpos[self.base_y_qpos_adr])
            self.data.qpos[self.base_x_qpos_adr] = new_x
            self.data.qpos[self.base_y_qpos_adr] = new_y
            self.data.qpos[self.base_yaw_qpos_adr] += dyaw
            ds_lin = float(-(new_x - prev_x) * np.sin(cur_yaw) + (new_y - prev_y) * np.cos(cur_yaw))
            half_gauge = 0.5 * TraxTrackAnimator.TRACK_WIDTH
            self.track_phase[0] += ds_lin - half_gauge * dyaw
            self.track_phase[1] += ds_lin + half_gauge * dyaw
            self.track_animator.update(self.track_phase)

        # Integrate the commanded setpoint, rather than the lagging measured
        # angle; otherwise actuator lag attenuates every velocity command.
        pan_target = self.data.ctrl[0] + safe_action[2] * dt * 1.5 * self.motor_gain
        tilt_target = self.data.ctrl[1] + safe_action[3] * dt * 1.5 * self.motor_gain

        # Purposeful skills use a comfort range, rotating the body before the
        # mechanical endpoint. Hard safety limits still apply independently.
        pan_limit = .90 if self.controller_mode == "goal_residual" else 1.047
        tilt_limit = .60 if self.controller_mode == "goal_residual" else .698
        self.data.ctrl[0] = float(np.clip(pan_target, -pan_limit, pan_limit))
        self.data.ctrl[1] = float(np.clip(tilt_target, -tilt_limit, tilt_limit))

        # Bước vật lý 100Hz
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)

        simulation_unstable = (not np.all(np.isfinite(self.data.qpos)) or
            not np.all(np.isfinite(self.data.qvel)) or
            any(self.data.warning[i].number > 0 for i in
                (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                 mujoco.mjtWarning.mjWARN_BADQACC)))

        # 5. Cập nhật Observation & Động lực nội tại
        obs = self._get_obs(action=safe_action)

        # Cập nhật các biến trạng thái thời gian (Temporal Dynamics Doc 03)
        if abs(safe_action[0]) > 0.05 or abs(safe_action[1]) > 0.05:
            self.time_since_base_motion = 0.0
        else:
            self.time_since_base_motion += dt

        if float(obs[18]) > 0.3:  # motion_score
            self.time_since_motion = 0.0
        else:
            self.time_since_motion += dt

        if float(obs[37]) > 0.3:  # object_novelty
            self.time_since_new_object = 0.0
        else:
            self.time_since_new_object += dt

        # Trích xuất độ mới từ compact state (Doc 13 Section 5.2: S0 không có novelty reward)
        if self.scenario_manager.scenario == BehavioralScenario.S0_MOTOR_SANITY:
            novelty_score = 0.0
            new_view_score = 0.0
        else:
            novelty_score = float(obs[36]) # local_region_novelty
            new_view_score = float(obs[35]) # current_view_novelty
        action_energy = float(np.sum(np.abs(safe_action[:4])))

        cube_p = self.data.xpos[self.obstacle_body_id] if self.obstacle_body_id != -1 else np.array([-0.5, 0.9, 0.04])
        nov_p = self.data.xpos[self.novel_body_id] if self.novel_body_id != -1 else np.array([0.6, 1.1, 0.06])
        is_colliding = bool(
            (self.obstacle_body_id != -1 and float(np.hypot(robot_x - cube_p[0], robot_y - cube_p[1])) <= 0.186) or
            (self.novel_body_id != -1 and float(np.hypot(robot_x - nov_p[0], robot_y - nov_p[1])) <= 0.132) or
            (abs(robot_x) >= 1.93 or abs(robot_y) >= 1.93)
        )
        if is_colliding:
            threat_score = 1.0

        is_resting = bool(action_energy < 0.08 and safe_action[4] < 0.2)
        self.drives.step(
            dt=dt,
            novelty_score=novelty_score,
            human_visible=self.human_agent.is_present,
            human_engaged=bool(self.human_agent.engagement_level > 0.5),
            action_energy=action_energy,
            is_resting=is_resting,
            threat_score=threat_score,
            sensor_loss=bool(goal_target is not None and not goal_target.visible)
        )

        # 6. Tính toán Composite Reward theo Doc 04 & Doc 13
        pan_norm = float(self.data.qpos[self.pan_qpos_adr if self.has_mobile_base else self.pan_joint_id] / 1.047)
        tilt_norm = float(self.data.qpos[self.tilt_qpos_adr if self.has_mobile_base else self.tilt_joint_id] / 0.698)
        target_is_visible = bool(has_active_target and target_dist < 3.5 and
            abs((target_bearing + current_pan + np.pi) % (2 * np.pi) - np.pi) < np.pi / 3)
        reward, reward_breakdown = self.reward_mixer.compute(
            action=safe_action,
            novelty_score=novelty_score,
            new_view_score=new_view_score,
            human_visible=self.human_agent.is_present,
            human_aligned=human_aligned,
            human_engagement=self.human_agent.engagement_level,
            front_obstacle_m=front_obstacle_m,
            is_colliding=is_colliding,
            is_spamming=is_spam,
            pan_norm=pan_norm,
            tilt_norm=tilt_norm,
            is_overridden=overridden,
            target_visible=target_is_visible,
            target_dist=target_dist,
            target_bearing=target_bearing,
            gaze_duration=self.scenario_manager.human_gaze_duration,
            pan_vel=float(safe_action[2]),
            is_behavior_switched=getattr(self.behavior_manager, "last_transition_switched", False),
            target_is_human=target_is_human,
            dt=dt
        )

        if self.controller_mode == "goal_residual":
            goal_reward = self.goal_manager.progress_delta * 4.0
            goal_reward += 1.0 if self.goal_manager.completion_event else 0.0
            goal_reward -= .25 if self.goal_manager.failure_event else 0.0
            reward += goal_reward
            reward_breakdown["reward/goal"] = float(goal_reward)
            reward_breakdown["reward/total"] = float(reward)
        if simulation_unstable:
            reward = -10.0
            reward_breakdown["reward/total"] = reward
            obs = np.nan_to_num(obs, nan=0, posinf=1, neginf=-1)
        terminated = simulation_unstable
        truncated = (self.step_count >= self.max_episode_steps)

        # Cập nhật Scenario Metrics (Doc 13 Section 2 & 13)
        scenario_metrics = self.scenario_manager.update_step(
            dt=dt,
            action=safe_action,
            safe_action=safe_action,
            overridden=overridden,
            override_reason=override_reason or "NONE",
            pan_deg=float(np.rad2deg(self.data.qpos[self.pan_qpos_adr if self.has_mobile_base else self.pan_joint_id]) + 90.0),
            tilt_deg=float(np.rad2deg(self.data.qpos[self.tilt_qpos_adr if self.has_mobile_base else self.tilt_joint_id]) + 90.0),
            target_visible=target_is_visible,
            target_bearing=target_bearing,
            target_distance=target_dist,
            human_aligned=human_aligned,
            human_responded=human_responded,
            tier=tier,
            is_spam=is_spam,
            novelty_score=novelty_score,
            target_id=goal_target.target_id if goal_target else (
                "human_01" if target_is_human else "ball_01" if self.has_rolling_ball else "static_object_01"),
            target_is_human=target_is_human,
            target_motion=bool(not target_is_human and self.has_rolling_ball and
                               (goal_target is None or goal_target.kind == "ball") and
                               np.linalg.norm(self.ball_vel[:2]) > .05),
            gaze_error=float((target_bearing + current_pan + np.pi) % (2 * np.pi) - np.pi),
            collision=is_colliding,
            simulation_unstable=simulation_unstable
        )

        safe_cmd = SafeCommand(
            linear_velocity=float(safe_action[0]),
            angular_velocity=float(safe_action[1]),
            pan_velocity=float(safe_action[2]),
            tilt_velocity=float(safe_action[3]),
            attention_intensity=float(safe_action[4]),
            overridden=overridden,
            override_reason=override_reason,
            safety_level="stop" if overridden and "COLLISION" in str(override_reason) else ("limited" if overridden else "normal")
        )

        # Multi-sector clearance telemetry (Upgrade Plan Section 9)
        clearances = self._get_multi_sector_clearance(robot_x, robot_y, robot_yaw)

        # Phân rã Reward rõ ràng thành 7 kênh chuẩn hóa (Upgrade Plan Section 3.3)
        canonical_reward = {
            "novelty": float(reward_breakdown.get("reward/novelty", 0.0) + reward_breakdown.get("reward/new_view", 0.0)),
            "exploration": float(reward_breakdown.get("reward/scan", 0.0) + reward_breakdown.get("reward/discovery", 0.0) + reward_breakdown.get("reward/dist_reduction", 0.0) + reward_breakdown.get("reward/fixation", 0.0) + reward_breakdown.get("reward/inspection", 0.0) + reward_breakdown.get("reward/approach", 0.0)),
            "social": float(reward_breakdown.get("reward/human", 0.0) + reward_breakdown.get("reward/engagement", 0.0) + reward_breakdown.get("reward/social_zone", 0.0) - reward_breakdown.get("penalty/signal_spam", 0.0) - reward_breakdown.get("penalty/crowding", 0.0)),
            "quiet": float(reward_breakdown.get("reward/quiet", 0.0)),
            "energy": float(-reward_breakdown.get("penalty/energy", 0.0) - reward_breakdown.get("penalty/excessive_movement", 0.0)),
            "smoothness": float(-reward_breakdown.get("penalty/jerk", 0.0) - reward_breakdown.get("penalty/repetition", 0.0) - reward_breakdown.get("penalty/head_oscillation", 0.0) - reward_breakdown.get("penalty/behavior_switch", 0.0)),
            "safety": float(-reward_breakdown.get("penalty/collision", 0.0) - reward_breakdown.get("penalty/near_obstacle", 0.0) - reward_breakdown.get("penalty/safety_override", 0.0) - reward_breakdown.get("penalty/servo_limit", 0.0) - reward_breakdown.get("penalty/reverse", 0.0)),
            "total": float(reward)
        }

        # Góc Yaw chuẩn hóa trong khoảng [-180 .. +180] độ và sin/cos (Upgrade Plan Section 3.2 & 8)
        wrapped_yaw = (robot_yaw + np.pi) % (2.0 * np.pi) - np.pi

        info = {
            "controller_mode": self.controller_mode,
            "policy_action": policy_action.tolist(),
            "controller_action": act_vec.tolist(),
            "goal_metrics": self.goal_manager.telemetry(),
            "simulation_unstable": simulation_unstable,
            "actuator_metrics": {
                "setpoint_clipped": bool(abs(pan_target) > pan_limit or abs(tilt_target) > tilt_limit),
                "pan_setpoint": float(self.data.ctrl[0]),
                "tilt_setpoint": float(self.data.ctrl[1]),
            },
            "reward_breakdown": reward_breakdown,
            "canonical_reward": canonical_reward,
            "internal_drives": self.drives.to_dict(),
            "novelty_metrics": self.novelty_engine.get_metrics(),
            "social_metrics": self.social_dynamics.get_social_state_dict(),
            "safety_metrics": self.behavior_manager.get_safety_metrics(),
            "scenario_metrics": scenario_metrics,
            "safe_command": safe_cmd,
            "scenario": self.scenario_manager.scenario.value,
            "ready_for_social_handoff": ready_for_handoff,
            "obstacle_clearance": float(front_obstacle_m),
            "clearances": clearances,
            "pose": {
                "x": float(robot_x),
                "y": float(robot_y),
                "yaw_deg": float(np.rad2deg(wrapped_yaw)),
                "yaw_rad": float(wrapped_yaw),
                "sin_yaw": float(np.sin(wrapped_yaw)),
                "cos_yaw": float(np.cos(wrapped_yaw))
            },
            "threat_score": float(threat_score),
            "step": self.step_count
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Khởi tạo hoặc cập nhật Passive MuJoCo 3D Viewer trực tiếp trên Main Thread.
        Hỗ trợ quan sát visual debugging thời gian thực khi huấn luyện hoặc kiểm tra mô hình.
        """
        if self.viewer is None:
            import mujoco.viewer
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 2.4
                self.viewer.cam.elevation = -22.0
                self.viewer.cam.azimuth = 145.0
                self.viewer.cam.lookat[:] = [0.0, 0.6, 0.15]
            except Exception as e:
                print(f"[Viewer Notice] Passive viewer launch notice: {e}")
                print("[Hint] Để mở giao diện đồ họa 3D trên macOS, hãy chạy qua: ./.venv/bin/mjpython")
                return None

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
