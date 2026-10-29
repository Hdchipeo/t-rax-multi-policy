import numpy as np
from typing import Dict, Any, Optional
from trax_rl.policies.base_policy import BasePolicy

class ThreatEscapePolicy(BasePolicy):
    """
    Threat Escape Policy (Upgrade Plan Section 7.4):
    Chuyên gia phản xạ thoát hiểm khẩn cấp khi mức độ đe dọa / Fear tăng vọt
    (va đập, vật cản lao sát hoặc kích thích âm thanh giật mình).
    
    Quy tắc điều khiển dưới ràng buộc 1 ToF phía trước:
    - Nếu Front ToF < 0.20m (quá sát): Kích hoạt lùi cứu nguy có kiểm soát ngắn.
    - Xoay nhanh 90 - 180 độ tại chỗ để đưa cảm biến ToF duy nhất sang hướng mở.
    - Thiết lập mức attention cao (Alert gesture) cảnh báo.
    """
    def __init__(self, name: str = "threat_escape_policy"):
        super().__init__(name)
        self.escape_phase = 0
        self.phase_timer = 0.0
        self.escape_turn_dir = 1.0

    def reset(self, turn_dir: Optional[float] = None):
        self.escape_phase = 0
        self.phase_timer = 0.0
        if turn_dir is not None and abs(turn_dir) > 0.01:
            self.escape_turn_dir = float(np.sign(turn_dir))
        else:
            self.escape_turn_dir = 1.0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # Cập nhật hướng xoay tất định theo vị trí vật cản nếu có telemetry
        if info is not None and "front_obstacle_x" in info:
            obs_x = float(info["front_obstacle_x"])
            if obs_x > 0.02:
                self.escape_turn_dir = 1.0   # Vật cản lệch phải -> xoay trái (+yaw)
            elif obs_x < -0.02:
                self.escape_turn_dir = -1.0  # Vật cản lệch trái -> xoay phải (-yaw)

        # Ưu tiên khoảng cách ToF thực (mét) do env truyền vào; fallback giải mã obs (schema v2: norm = d / 2.5)
        if info is not None and "front_tof_m" in info:
            front_tof_m = float(info["front_tof_m"])
        else:
            front_tof_norm = float(obs[6]) if (obs is not None and len(obs) > 6) else 1.0
            front_tof_m = front_tof_norm * 2.5

        lin_vel = 0.0
        ang_vel = 0.0
        pan_vel = 0.0
        tilt_vel = 0.0
        attn = 0.85  # Alert gesture (Tier 3)

        if front_tof_m < 0.20:
            # Giai đoạn 1: Lùi khẩn cấp để tạo khoảng trống thoát hiểm
            lin_vel = -0.15
            ang_vel = self.escape_turn_dir * 0.40
        else:
            # Giai đoạn 2: Xoay vòng dứt khoát tìm hướng thoáng
            lin_vel = 0.05 if front_tof_m > 0.40 else 0.0
            ang_vel = self.escape_turn_dir * 1.10

        return np.array([lin_vel, ang_vel, pan_vel, tilt_vel, attn], dtype=np.float32)
