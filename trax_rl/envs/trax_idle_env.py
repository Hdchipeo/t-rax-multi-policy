import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

from trax_rl.observations.schemas import build_idle_observation
from trax_rl.rewards.reward_mixer import IdleRewardMixer

class TraxIdleEnv(gym.Env):
    """
    Gymnasium Environment cho Autonomous Idle Policy trên T-RAX Robot.
    Sử dụng MuJoCo C bindings chính thức để mô phỏng vật lý chính xác cao.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, xml_path=None, frameskip=10, max_episode_steps=500):
        super().__init__()
        if xml_path is None:
            # Đường dẫn tương đối từ project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            xml_path = os.path.join(base_dir, "assets", "trax", "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frameskip = frameskip
        self.max_episode_steps = max_episode_steps

        # Action: 5 Discrete actions
        # 0: REST, 1: LOOK_LEFT, 2: LOOK_RIGHT, 3: LOOK_CENTER, 4: TRACK_TARGET
        self.action_space = spaces.Discrete(5)

        # Observation: Box 6-dim [-1.0 .. 1.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.reward_mixer = IdleRewardMixer()
        self.step_count = 0
        self.prev_action = 0
        self.time_since_target_seen = 0.0
        self.target_present_in_episode = True

        # Joint & actuator IDs
        self.pan_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_pan")
        self.tilt_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_tilt")
        self.target_mocap_id = self.model.body("target").mocapid[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.prev_action = 0
        self.time_since_target_seen = 0.0

        # Domain Randomization: Xác suất mục tiêu xuất hiện trong phòng
        self.target_present_in_episode = (self.np_random.uniform() > 0.3)
        self.target_speed = self.np_random.uniform(0.3, 1.0)
        self.target_phase = self.np_random.uniform(0, 2 * np.pi)

        # Đặt vị trí ban đầu cho mục tiêu qua mocap_pos
        if self.target_present_in_episode:
            self.data.mocap_pos[self.target_mocap_id] = np.array([
                float(np.sin(self.target_phase) * 0.8),
                float(1.2 + np.cos(self.target_phase) * 0.3),
                0.35
            ])
        else:
            # Ẩn mục tiêu ra xa tầm nhìn
            self.data.mocap_pos[self.target_mocap_id] = np.array([5.0, 5.0, 0.35])

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        pan_rad = self.data.qpos[self.pan_joint_id]
        max_pan = 1.047  # ~60 độ

        # Lấy tọa độ mục tiêu từ mocap
        target_pos_x = self.data.mocap_pos[self.target_mocap_id][0]
        target_pos_y = self.data.mocap_pos[self.target_mocap_id][1]

        # Kiểm tra mục tiêu có nằm trong trường nhìn (FOV ~ 60 độ)
        distance = np.sqrt(target_pos_x**2 + target_pos_y**2)
        angle_to_target = np.arctan2(target_pos_x, target_pos_y)

        # Lệch góc so với hướng nhìn của đầu:
        # Trong hệ MuJoCo (Z-up), góc quay dương quanh Z là sang TRÁI, góc âm là sang PHẢI.
        # Khi đầu thẳng (pan_rad = 0) và bóng ở X > 0 (bên phải): bóng nằm ở nửa phải của ảnh (relative_angle > 0).
        # Khi đầu quay bám bóng (pan_rad = -angle_to_target): relative_angle = 0 (bóng nằm chính giữa tâm ảnh).
        relative_angle = angle_to_target + pan_rad
        target_visible = (distance < 3.0) and (abs(relative_angle) < (np.pi / 3.0))

        if target_visible:
            self.time_since_target_seen = 0.0
            target_x_norm = float(np.clip(relative_angle / (np.pi / 6.0), -1.0, 1.0))
            motion_level = 0.5
        else:
            self.time_since_target_seen += 0.1
            target_x_norm = 0.0
            motion_level = 0.0

        return build_idle_observation(
            target_visible=target_visible,
            target_x_norm=target_x_norm,
            motion_level=motion_level,
            head_pan_rad=pan_rad,
            max_pan_rad=max_pan,
            time_since_event_sec=self.time_since_target_seen,
            prev_action_id=self.prev_action
        )

    def step(self, action):
        self.step_count += 1

        # Chuyển đổi Discrete Primitive Action thành setpoint góc servo (2 trục Pan & Tilt)
        pan_target = self.data.qpos[self.pan_joint_id]
        tilt_target = self.data.qpos[self.tilt_joint_id]
        energy_cost = 0.0

        if action == 0:    # REST (Nghỉ ngơi - Đầu hướng tâm và hơi cúi ngủ nhẹ)
            pan_target = 0.0
            tilt_target = -0.18  # Cúi đầu nhẹ ~ -10°
            energy_cost = 0.02
        elif action == 1:  # LOOK_LEFT
            pan_target = min(1.0, pan_target + 0.3)
            tilt_target = 0.05   # Ngẩng nhẹ quan sát
            energy_cost = 0.2
        elif action == 2:  # LOOK_RIGHT
            pan_target = max(-1.0, pan_target - 0.3)
            tilt_target = 0.05
            energy_cost = 0.2
        elif action == 3:  # LOOK_CENTER
            pan_target = 0.0
            tilt_target = 0.0
            energy_cost = 0.1
        elif action == 4:  # TRACK_TARGET (Bám mục tiêu theo cả 2 trục Pan và Tilt)
            target_pos_x = self.data.mocap_pos[self.target_mocap_id][0]
            target_pos_y = self.data.mocap_pos[self.target_mocap_id][1]
            target_pos_z = self.data.mocap_pos[self.target_mocap_id][2]

            # 1. Trục Yaw (Pan - Xoay ngang):
            # Lưu ý hệ MuJoCo (Z-up): Chiều quay dương là sang TRÁI (ngược chiều kim đồng hồ).
            # Do đó khi bóng ở X > 0 (bên phải), pan_target phải mang dấu ÂM (-) để đầu quay sang PHẢI bám bóng!
            angle_to_target = -np.arctan2(target_pos_x, target_pos_y)
            pan_target = float(np.clip(angle_to_target, -1.047, 1.047))

            # 2. Trục Pitch (Tilt - Gật dọc): Hướng góc nhìn camera thẳng vào mục tiêu
            dist_xy = np.sqrt(target_pos_x**2 + target_pos_y**2)
            delta_z = target_pos_z - 0.06  # Vị trí trục tilt khớp cổ Z ~ 0.06m
            tilt_target = float(np.clip(np.arctan2(delta_z, dist_xy), -0.698, 0.698))
            energy_cost = 0.25

        # Điều khiển tín hiệu PWM đến cả 2 servo actuators
        self.data.ctrl[0] = pan_target
        self.data.ctrl[1] = tilt_target

        # Di chuyển mục tiêu theo quỹ đạo hình sin tự nhiên
        if self.target_present_in_episode:
            t = self.step_count * 0.1 * self.target_speed + self.target_phase
            self.data.mocap_pos[self.target_mocap_id][0] = float(np.sin(t) * 0.9)

        # Stepping physics (10hz control loop)
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self.reward_mixer.compute(obs, action, self.prev_action, energy_cost)
        self.prev_action = action

        terminated = False
        truncated = (self.step_count >= self.max_episode_steps)
        info = {
            "visible": bool(obs[0] > 0.5),
            "tracked": bool(obs[0] > 0.5 and abs(obs[1]) < 0.35 and action == 4),
            "is_success": bool(obs[0] > 0.5 and abs(obs[1]) < 0.2 and action == 4),
            "step": self.step_count
        }

        return obs, reward, terminated, truncated, info
