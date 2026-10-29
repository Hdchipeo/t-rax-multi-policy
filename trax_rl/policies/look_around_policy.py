import numpy as np
from typing import Dict, Any, Optional
from trax_rl.policies.base_policy import BasePolicy

class LookAroundPolicy(BasePolicy):
    """
    Look Around / Perception Policy (Upgrade Plan Phase 4):
    Chuyên gia điều khiển 2 trục servo Pan/Tilt đầu robot để quét góc nhìn, tìm kiếm
    kích thích mới và giải tỏa boredom mà không cần di chuyển bánh xích (tiết kiệm pin & tĩnh lặng cơ học).
    """
    def __init__(self, name: str = "look_around_policy"):
        super().__init__(name)
        self.scan_phase = 0.0
        self.direction = 1.0

    def reset(self):
        self.scan_phase = 0.0
        self.direction = 1.0 if np.random.rand() > 0.5 else -1.0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        pan_norm = float(obs[0]) if len(obs) > 0 else 0.0
        tilt_norm = float(obs[1]) if len(obs) > 1 else 0.0

        # Đảo chiều quét khi chạm giới hạn servo
        if pan_norm >= 0.75:
            self.direction = -1.0
        elif pan_norm <= -0.75:
            self.direction = 1.0

        pan_vel = float(self.direction * 0.70)
        # Gật nhẹ đầu lên xuống theo nhịp điều hòa
        self.scan_phase += 0.1
        tilt_vel = float(0.35 * np.cos(self.scan_phase * 0.8))

        # Giữ nguyên đế bánh xích tuyệt đối để tĩnh lặng cơ học
        lin_vel = 0.0
        ang_vel = 0.0
        attn = 0.20

        return np.array([lin_vel, ang_vel, pan_vel, tilt_vel, attn], dtype=np.float32)
