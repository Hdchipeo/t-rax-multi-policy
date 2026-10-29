import numpy as np
from typing import Dict, Any, Optional

def build_behavioral_observation_dict(
    # Robot (8)
    head_pan_norm: float,
    head_tilt_norm: float,
    linear_velocity_norm: float,
    angular_velocity_norm: float,
    left_motor_velocity_norm: float,
    right_motor_velocity_norm: float,
    front_distance_norm: float,
    battery_norm: float,
    # Vision (16)
    face_visible: bool,
    face_x: float,
    face_y: float,
    face_size: float,
    face_confidence: float,
    object_visible: bool,
    object_x: float,
    object_y: float,
    object_size: float,
    object_confidence: float,
    motion_score: float,
    scene_change_score: float,
    object_embedding: np.ndarray,
    # Temporal (6)
    time_since_new_object: float,
    time_since_human_seen: float,
    time_since_human_response: float,
    time_since_motion: float,
    time_since_attention_signal: float,
    time_since_base_motion: float,
    # Internal (5)
    curiosity: float,
    boredom: float,
    social_need: float,
    energy_budget: float,
    perception_confidence: float,
    # Exploration (4)
    current_view_novelty: float,
    local_region_novelty: float,
    object_novelty: float,
    recent_action_diversity: float
) -> Dict[str, np.ndarray]:
    """
    T-RAX Behavioral Observation Schema V1 (Doc 03 - Observation & Action Specification).
    Tổng cộng 39 scalar features được phân nhóm có ngữ nghĩa.
    """
    # 1. Robot features (8)
    robot_vec = np.array([
        np.clip(head_pan_norm, -1.0, 1.0),
        np.clip(head_tilt_norm, -1.0, 1.0),
        np.clip(linear_velocity_norm, -1.0, 1.0),
        np.clip(angular_velocity_norm, -1.0, 1.0),
        np.clip(left_motor_velocity_norm, -1.0, 1.0),
        np.clip(right_motor_velocity_norm, -1.0, 1.0),
        np.clip(front_distance_norm, 0.0, 1.0),
        np.clip(battery_norm, 0.0, 1.0)
    ], dtype=np.float32)

    # 2. Vision features (16)
    emb = np.zeros(4, dtype=np.float32)
    if object_embedding is not None and len(object_embedding) == 4:
        emb = np.clip(np.asarray(object_embedding, dtype=np.float32), -1.0, 1.0)

    vision_vec = np.array([
        1.0 if face_visible else 0.0,
        np.clip(face_x, -1.0, 1.0),
        np.clip(face_y, -1.0, 1.0),
        np.clip(face_size, 0.0, 1.0),
        np.clip(face_confidence, 0.0, 1.0),
        1.0 if object_visible else 0.0,
        np.clip(object_x, -1.0, 1.0),
        np.clip(object_y, -1.0, 1.0),
        np.clip(object_size, 0.0, 1.0),
        np.clip(object_confidence, 0.0, 1.0),
        np.clip(motion_score, 0.0, 1.0),
        np.clip(scene_change_score, 0.0, 1.0),
        emb[0], emb[1], emb[2], emb[3]
    ], dtype=np.float32)

    # 3. Temporal features (6) - Chuẩn hóa chia cho 10.0 giây
    temporal_vec = np.array([
        np.clip(time_since_new_object / 10.0, 0.0, 1.0),
        np.clip(time_since_human_seen / 10.0, 0.0, 1.0),
        np.clip(time_since_human_response / 10.0, 0.0, 1.0),
        np.clip(time_since_motion / 10.0, 0.0, 1.0),
        np.clip(time_since_attention_signal / 10.0, 0.0, 1.0),
        np.clip(time_since_base_motion / 10.0, 0.0, 1.0)
    ], dtype=np.float32)

    # 4. Internal drives features (5)
    internal_vec = np.array([
        np.clip(curiosity, 0.0, 1.0),
        np.clip(boredom, 0.0, 1.0),
        np.clip(social_need, 0.0, 1.0),
        np.clip(energy_budget, 0.0, 1.0),
        np.clip(perception_confidence, 0.0, 1.0)
    ], dtype=np.float32)

    # 5. Exploration features (4)
    exploration_vec = np.array([
        np.clip(current_view_novelty, 0.0, 1.0),
        np.clip(local_region_novelty, 0.0, 1.0),
        np.clip(object_novelty, 0.0, 1.0),
        np.clip(recent_action_diversity, 0.0, 1.0)
    ], dtype=np.float32)

    return {
        "robot": robot_vec,
        "vision": vision_vec,
        "temporal": temporal_vec,
        "internal": internal_vec,
        "exploration": exploration_vec
    }

def flatten_behavioral_observation(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Ghép các vector thành 1 mảng numpy duy nhất 39 chiều cho mạng nơ-ron:
    8 + 16 + 6 + 5 + 4 = 39 float32
    """
    return np.concatenate([
        obs_dict["robot"],
        obs_dict["vision"],
        obs_dict["temporal"],
        obs_dict["internal"],
        obs_dict["exploration"]
    ]).astype(np.float32)

def build_robot_centric_observation(
    linear_velocity: float,
    angular_velocity: float,
    robot_yaw_rad: float,
    pan_norm: float,
    tilt_norm: float,
    target_visible: bool,
    target_distance: float,
    target_bearing_rad: float,
    front_tof_distance: float,
    drives: Dict[str, float],
    prev_action: np.ndarray,
    track_half_width: float = 0.07,
    clearances: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    T-RAX Robot-Centric Observation Schema (Hardware Grounded: 1 ToF Sensor).
    Khớp chính xác với phần cứng robot T-RAX thực tế (ESP32):
      - 2 động cơ xích (differential drive telemetry)
      - 2 servo Pan / Tilt đầu robot
      - 1 Camera nhận diện đối tượng / người (Visual Bearing & Relative Dist)
      - DUY NHẤT 1 cảm biến ToF phía trước (Front ToF Range: 0.05m .. 2.5m)
      - Không sử dụng cảm biến bên hông hoặc phía sau (Vùng mù thực tế)
      - Mã hóa sin/cos góc quay yaw và góc lệch mục tiêu để tránh gián đoạn biên +-180 độ.
    """
    wrapped_yaw = (robot_yaw_rad + np.pi) % (2.0 * np.pi) - np.pi
    sin_yaw = float(np.sin(wrapped_yaw))
    cos_yaw = float(np.cos(wrapped_yaw))

    sin_bearing = float(np.sin(target_bearing_rad)) if target_visible else 0.0
    cos_bearing = float(np.cos(target_bearing_rad)) if target_visible else 0.0

    prev_lin = float(prev_action[0]) if len(prev_action) > 0 else 0.0
    prev_ang = float(prev_action[1]) if len(prev_action) > 1 else 0.0
    prev_left = prev_lin - prev_ang * track_half_width
    prev_right = prev_lin + prev_ang * track_half_width

    # Giá trị đo từ duy nhất 1 cảm biến ToF phía trước (chuẩn hóa trên tầm đo tối đa 2.5m)
    c_front_norm = float(np.clip(front_tof_distance / 2.5, 0.0, 1.0))

    curiosity = float(drives.get("curiosity", 0.5))
    boredom = float(drives.get("boredom", 0.1))
    fear = float(drives.get("fear", 0.0))
    safety = float(drives.get("safety", 1.0 - fear))
    arousal = float(drives.get("arousal", 0.2))

    # Feature vector chuẩn hóa hoàn toàn khả thi trên ESP32 runtime
    features = np.array([
        np.clip(linear_velocity, -1.0, 1.0),
        np.clip(angular_velocity, -1.0, 1.0),
        sin_yaw,
        cos_yaw,
        np.clip(pan_norm, -1.0, 1.0),
        np.clip(tilt_norm, -1.0, 1.0),
        1.0 if target_visible else 0.0,
        float(np.clip(target_distance / 3.5, 0.0, 1.0)),
        sin_bearing,
        cos_bearing,
        c_front_norm, # Duy nhất 1 cảm biến ToF phía trước
        curiosity,
        boredom,
        fear,
        safety,
        arousal,
        np.clip(prev_left, -1.0, 1.0),
        np.clip(prev_right, -1.0, 1.0)
    ], dtype=np.float32)

    ret = {
        "features": features,
        "wrapped_yaw_rad": wrapped_yaw,
        "sin_yaw": sin_yaw,
        "cos_yaw": cos_yaw,
        "front_tof_m": float(front_tof_distance),
        "front_tof_norm": c_front_norm,
        "target_relative": {
            "visible": target_visible,
            "distance": float(target_distance),
            "bearing_rad": float(target_bearing_rad),
            "sin_bearing": sin_bearing,
            "cos_bearing": cos_bearing
        }
    }
    if clearances is not None:
        ret["sim_ground_truth_clearances"] = clearances

    return ret
