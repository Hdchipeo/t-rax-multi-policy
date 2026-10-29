#!/usr/bin/env python3
"""
T-RAX MuJoCo Real-time Interactive 3D Viewer & Policy Visualizer
Supports:
1. Running the trained Autonomous Idle Policy (Discrete 5 actions, 6-dim obs).
2. Running the Autonomous Behavioral Dynamics Environment (Continuous 5D, 39-dim obs, 5 Internal Drives).
3. Interactive manual physics inspection and joint manipulation (Standalone UI).
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_idle_env import TraxIdleEnv
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv
from trax_rl.evaluation import environment_options

ACTION_NAMES = {
    0: "REST",
    1: "LOOK_LEFT",
    2: "LOOK_RIGHT",
    3: "LOOK_CENTER",
    4: "TRACK_TARGET"
}

ATTENTION_TIER_NAMES = {
    0: "Tier 0 (Quiet)",
    1: "Tier 1 (Micro-motion)",
    2: "Tier 2 (Head Gesture)",
    3: "Tier 3 (Alert Gesture)",
    4: "Tier 4 (Audio Chirp)"
}

def run_policy_viewer(model_path: str = "checkpoints/purposeful_behavioral_v2/final_model.zip",
                      env_type: str = "purposeful",
                      speed_factor: float = 1.0,
                      scenario: str = "S7",
                      deterministic: bool = True,
                      log_style: str = "compact"):
    print("==================================================================")
    print("🤖 T-RAX MUJOCO REAL-TIME 3D VISUALIZER")
    print(f"🌍 Environment      : {env_type.upper()} [Scenario: {scenario.upper()}]")
    print(f"📦 Model Checkpoint : {model_path}")
    print(f"⚡ Playback Speed   : {speed_factor}x | Log Style: {log_style.upper()}")
    print("🎮 Controls         : Chuột trái xoay, chuột phải dịch, cuộn phóng to")
    print("                      Ctrl + chuột phải để kéo vật thể trong không gian")
    print("                      Space: Tạm dừng / Tiếp tục | ESC: Thoát")
    print("==================================================================")

    # 1. Khởi tạo môi trường theo tùy chọn
    if env_type in ("behavioral", "purposeful"):
        metadata_path = Path(model_path).with_suffix(".metadata.json") if model_path else None
        config = None
        if metadata_path and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("controller_mode") == "goal_residual":
                env_type = "purposeful"
                config = metadata["config"]
        if env_type == "purposeful":
            config = config or yaml.safe_load(Path("configs/policies/purposeful_v2.yaml").read_text())
            env = TraxBehavioralEnv(scenario=scenario, **environment_options(config))
        else:
            env = TraxBehavioralEnv(action_mode="continuous", scenario=scenario)
        obs_dim = env.observation_space.shape[0]
    else:
        env = TraxIdleEnv()
        obs_dim = 6

    # 2. Tải Policy nếu file tồn tại
    has_policy = model_path is not None and os.path.exists(model_path)
    model = None
    if has_policy:
        print(f"✅ Loading trained policy from: {model_path}")
        try:
            model = PPO.load(model_path, device="cpu")
        except Exception:
            model = SAC.load(model_path, device="cpu")
    else:
        if model_path:
            env.close()
            raise FileNotFoundError(model_path)
        print("Running nominal controller demo without learned residual.")
    if model is not None and model.observation_space.shape != (obs_dim,):
        env.close()
        raise ValueError("Checkpoint and viewer environment have different observation schemas")

    obs, info = env.reset()
    episode_idx = 1
    total_reward = 0.0
    step_idx = 0

    # 3. Khởi chạy MuJoCo Passive Viewer trên Main Thread
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 145.0
        viewer.cam.lookat[:] = [0.0, 0.5, 0.15]

        dt_target = 0.1 / max(0.1, speed_factor)  # 10Hz control loop

        while viewer.is_running():
            step_start = time.time()

            # Lựa chọn hành động từ policy hoặc heuristic
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
                if env_type == "idle":
                    action = int(action)
            else:
                if env_type == "purposeful":
                    action = np.zeros(5, dtype=np.float32)
                elif env_type == "behavioral":
                    # Heuristic behavioral action [lin, ang, pan_vel, tilt_vel, attn]
                    pan_adr = env.pan_qpos_adr if hasattr(env, "pan_qpos_adr") else env.pan_joint_id
                    tilt_adr = env.tilt_qpos_adr if hasattr(env, "tilt_qpos_adr") else env.tilt_joint_id
                    pan_cur = float(env.data.qpos[pan_adr])
                    tilt_cur = float(env.data.qpos[tilt_adr])

                    robot_x = float(env.data.qpos[env.base_x_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0
                    robot_y = float(env.data.qpos[env.base_y_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0
                    robot_yaw = float(env.data.qpos[env.base_yaw_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0

                    tgt_w = env.data.mocap_pos[env.target_mocap_id]
                    dx_w = tgt_w[0] - robot_x
                    dy_w = tgt_w[1] - robot_y
                    dx_rob = dx_w * np.cos(robot_yaw) + dy_w * np.sin(robot_yaw)
                    dy_rob = -dx_w * np.sin(robot_yaw) + dy_w * np.cos(robot_yaw)

                    target_pan = -float(np.arctan2(dx_rob, dy_rob))
                    dist_xy = float(np.sqrt(dx_rob**2 + dy_rob**2))
                    target_tilt = float(np.arctan2(tgt_w[2] - 0.06, dist_xy))

                    pan_vel = float(np.clip((target_pan - pan_cur) * 2.0, -1.0, 1.0))
                    tilt_vel = float(np.clip((target_tilt - tilt_cur) * 2.0, -1.0, 1.0))
                    lin_vel = float(np.clip((dist_xy - 0.8) * 0.4, 0.0, 0.3)) if dist_xy > 0.8 else 0.0
                    ang_vel = float(np.clip(-dx_rob * 1.2, -0.6, 0.6))

                    action = np.array([lin_vel, ang_vel, pan_vel, tilt_vel, 0.35], dtype=np.float32)
                else:
                    action = 4 if obs[0] > 0.5 else 0

            # Bước vật lý 10Hz
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_idx += 1

            viewer.sync()

            pan_adr = env.pan_qpos_adr if hasattr(env, "pan_qpos_adr") else env.pan_joint_id
            tilt_adr = env.tilt_qpos_adr if hasattr(env, "tilt_qpos_adr") else env.tilt_joint_id
            pan_deg = np.rad2deg(env.data.qpos[pan_adr]) + 90.0
            tilt_deg = np.rad2deg(env.data.qpos[tilt_adr]) + 90.0
            target_x = env.data.mocap_pos[env.target_mocap_id][0]

            if step_idx % 10 == 0:
                if env_type in ("behavioral", "purposeful"):
                    drives = info.get("internal_drives", {})
                    safety_met = info.get("safety_metrics", {})
                    canon_rwd = info.get("canonical_reward", {})
                    social = info.get("social_metrics", {})

                    c_val = drives.get("curiosity", 0.0)
                    b_val = drives.get("boredom", 0.0)
                    f_val = drives.get("fear", 0.0)
                    safe_val = drives.get("safety", 1.0 - f_val)
                    a_val = drives.get("arousal", 0.20)
                    nov_val = canon_rwd.get("novelty", 0.0)

                    robot_x = float(env.data.qpos[env.base_x_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0
                    robot_y = float(env.data.qpos[env.base_y_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0
                    robot_yaw = float(env.data.qpos[env.base_yaw_qpos_adr]) if getattr(env, "has_mobile_base", False) else 0.0
                    # Wrapped Yaw [-180 .. +180] deg
                    yaw_deg = float(np.rad2deg((robot_yaw + np.pi) % (2.0 * np.pi) - np.pi))

                    safe_cmd = info.get("safe_command")
                    goal = info.get("goal_metrics", {})
                    goal_tag = f" | Goal: {goal.get('type')}/{goal.get('phase')} ({goal.get('target_id')})" if env_type == "purposeful" else ""
                    safe_tag = f" 🛑 [{safe_cmd.override_reason}]" if safe_cmd and safe_cmd.overridden else ""

                    target_p = env.data.mocap_pos[env.target_mocap_id]
                    clr_val = float(info.get("obstacle_clearance", 2.5))
                    beh_name = safety_met.get("current_behavior", "IDLE")
                    trig_name = safety_met.get("trigger", "NONE")
                    act_policy = safety_met.get("active_policy", "autonomous_idle_policy")
                    intervention = safety_met.get("intervention", "NONE")
                    raw_l = safety_met.get("raw_left", 0.0)
                    raw_r = safety_met.get("raw_right", 0.0)
                    safe_l = safety_met.get("safe_left", 0.0)
                    safe_r = safety_met.get("safe_right", 0.0)

                    if log_style == "detailed":
                        # Định dạng chi tiết theo Upgrade Plan Section 14
                        print(f"\n[Episode {episode_idx:02d} | Step {step_idx:03d}]")
                        print(f"State:")
                        print(f"  pose        = ({robot_x:+.2f}, {robot_y:+.2f})")
                        print(f"  yaw         = {yaw_deg:+.0f} deg")
                        print(f"  velocity    = {safe_cmd.linear_velocity:.2f} m/s")
                        print(f"Head:")
                        print(f"  pan         = {pan_deg:.0f} deg")
                        print(f"  tilt        = {tilt_deg:.0f} deg")
                        print(f"Perception:")
                        print(f"  object_visible = {bool(obs[13] > 0.5)}")
                        print(f"  ball_distance  = {clr_val:.2f} m")
                        print(f"  novelty        = {nov_val:.2f}")
                        print(f"Drives:")
                        print(f"  curiosity = {c_val:.2f}")
                        print(f"  boredom   = {b_val:.2f}")
                        print(f"  fear      = {f_val:.2f}")
                        print(f"  safety    = {safe_val:.2f}")
                        print(f"  arousal   = {a_val:.2f}")
                        print(f"Behavior:")
                        print(f"  selected = {beh_name}")
                        print(f"  trigger  = {trig_name}")
                        if env_type == "purposeful":
                            print(f"  goal     = {goal}")
                        print(f"Policy:")
                        print(f"  active   = {act_policy}")
                        print(f"Action:")
                        print(f"  raw_left   = {raw_l:+.2f}")
                        print(f"  raw_right  = {raw_r:+.2f}")
                        print(f"  safe_left  = {safe_l:+.2f}")
                        print(f"  safe_right = {safe_r:+.2f}")
                        print(f"Safety:")
                        print(f"  intervention = {intervention}")
                        print(f"Reward:")
                        print(f"  novelty           = {canon_rwd.get('novelty', 0.0):+.2f}")
                        print(f"  exploration       = {canon_rwd.get('exploration', 0.0):+.2f}")
                        print(f"  object_interest   = {canon_rwd.get('object_interest', 0.0):+.2f}")
                        print(f"  smooth_motion     = {canon_rwd.get('smooth_motion', 0.0):+.2f}")
                        print(f"  near_obstacle     = {canon_rwd.get('near_obstacle', 0.0):+.2f}")
                        print(f"  energy_cost       = {canon_rwd.get('energy_cost', 0.0):+.2f}")
                        print(f"  safety_penalty    = {canon_rwd.get('safety_intervention', 0.0):+.2f}")
                        print(f"Total:")
                        print(f"  {reward:+.2f}")
                    else:
                        # Định dạng compact 1 dòng rõ ràng, chuẩn hóa góc quay và động lực nội tại
                        print(f"[Ep {episode_idx:02d} | Step {step_idx:03d}] "
                              f"Base: ({robot_x:+4.2f}, {robot_y:+4.2f}) ψ:{yaw_deg:+4.0f}° | "
                              f"Head: [P:{pan_deg:3.0f}° T:{tilt_deg:3.0f}°] | "
                              f"Ball: ({target_p[0]:+4.2f}, {target_p[1]:+4.2f}, {target_p[2]:4.2f}) | "
                              f"Clr: {clr_val:4.2f}m | "
                              f"Drives: [C:{c_val:.2f} B:{b_val:.2f} F:{f_val:.2f} A:{a_val:.2f}] | "
                              f"Beh: {beh_name:<7} ({trig_name:<16}) | "
                              f"Rwd: {reward:+5.2f}{safe_tag}{goal_tag}")
                else:
                    act_name = ACTION_NAMES.get(action, "UNKNOWN")
                    print(f"[Ep {episode_idx:02d} | Step {step_idx:03d}] Act: {act_name:<12} | "
                          f"Pan: {pan_deg:5.1f}° | Tilt: {tilt_deg:5.1f}° | Target X: {target_x:+5.2f}m | "
                          f"Reward: {reward:+6.2f} (Total: {total_reward:6.1f})")

            # Xử lý kết thúc episode
            if terminated or truncated:
                print(f"🏁 Episode {episode_idx} finished! Total Reward: {total_reward:.2f}\n")
                obs, info = env.reset()
                episode_idx += 1
                total_reward = 0.0
                step_idx = 0

            # Giữ nhịp thời gian thực 10Hz
            elapsed = time.time() - step_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

def run_standalone_viewer(scene_xml: str = "assets/trax/scene.xml"):
    print(f"Launching Native MuJoCo Standalone UI Viewer for: {scene_xml} ...")
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 145.0
        viewer.cam.lookat[:] = [0.0, 0.3, 0.1]
        print("✅ MuJoCo Viewer is running! Press ESC or close window to exit.")
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def main():
    parser = argparse.ArgumentParser(description="T-RAX MuJoCo 3D Interactive UI & Policy Visualizer")
    parser.add_argument("--mode", choices=["policy", "standalone"], default="policy",
                        help="'policy': Chạy mô hình RL; 'standalone': Mở giao diện MuJoCo gốc để chỉnh joint sliders")
    parser.add_argument("--env", choices=["idle", "behavioral", "purposeful"], default="purposeful",
                        help="'behavioral': Môi trường Autonomous Behavioral Dynamics Phase 4; 'idle': Môi trường Idle v0.1")
    parser.add_argument("--scenario", type=str, default="S7", choices=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "mixed"],
                        help="Kịch bản hành vi S0-S7 theo tài liệu 13_behavioral_scenarios.md")
    parser.add_argument("--model", type=str, default=None,
                        help="Đường dẫn tới file trọng số SB3 (.zip)")
    parser.add_argument("--nominal", action="store_true", help="Run without learned policy; explicit baseline demo")
    parser.add_argument("--scene", type=str, default="assets/trax/scene.xml",
                        help="Đường dẫn tới file MJCF scene.xml")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Hệ số tốc độ mô phỏng (1.0 = thời gian thực 10Hz)")
    parser.add_argument("--log-style", choices=["compact", "detailed"], default="compact",
                        help="'compact': Log 1 dòng tóm tắt chuẩn hóa; 'detailed': Log từng khối chi tiết giải thích theo Upgrade Plan Section 14")
    args = parser.parse_args()

    if args.mode == "standalone":
        run_standalone_viewer(args.scene)
    else:
        defaults = {"purposeful": "purposeful_behavioral_v2", "behavioral": "autonomous_behavioral_dynamics", "idle": "autonomous_idle"}
        path = args.model or f"checkpoints/{defaults[args.env]}/final_model.zip"
        run_policy_viewer(None if args.nominal else path, env_type=args.env, speed_factor=args.speed, scenario=args.scenario, log_style=args.log_style)

if __name__ == "__main__":
    main()
