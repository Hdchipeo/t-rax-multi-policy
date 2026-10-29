import numpy as np
from typing import Dict, Any, Optional
from trax_rl.policies.base_policy import BasePolicy

class ExplorePolicy(BasePolicy):
    """
    Explore Policy (Upgrade Plan Phase 3):
    Chuyên gia điều khiển chuyển động bánh xích để khám phá an toàn không gian
    dưới ràng buộc phần cứng: DUY NHẤT 1 CẢM BIẾN ToF PHÍA TRƯỚC.
    
    Nguyên lý:
    - Khi phía trước thông thoáng (Front ToF > 0.6m): Duy trì tốc độ tiến mượt mà kèm góc lượn nhẹ
      để tối đa hóa độ bao phủ diện tích (area coverage).
    - Khi phát hiện vật cản từ xa (0.3m <= Front ToF <= 0.6m): Giảm tốc độ tuyến tính và bắt đầu bẻ lái né tránh.
    - Khi vật cản cự ly gần (< 0.25m): Dừng tiến hoàn toàn, quay tại chỗ dứt khoát theo hướng mở.
    """
    def __init__(self, name: str = "explore_policy"):
        super().__init__(name)
        self.turn_direction = 1.0
        self.turn_timer = 0.0

    def reset(self):
        self.turn_direction = 1.0 if np.random.rand() > 0.5 else -1.0
        self.turn_timer = 0.0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # Trích xuất dữ liệu cảm biến ToF duy nhất phía trước
        # obs[6] = front_distance_norm trong observation 39-dim
        front_tof_norm = float(obs[6]) if len(obs) > 6 else 1.0
        front_tof_m = front_tof_norm * 3.0  # Chuẩn hóa về mét

        # Trích xuất động lực nội tại
        curiosity = float(obs[30]) if len(obs) > 30 else 0.5
        boredom = float(obs[31]) if len(obs) > 31 else 0.1

        lin_vel = 0.0
        ang_vel = 0.0
        pan_vel = 0.0
        tilt_vel = 0.0
        attn = 0.10

        if self.turn_timer > 0.0:
            self.turn_timer -= 0.1
            lin_vel = 0.0
            ang_vel = self.turn_direction * 0.75
        elif front_tof_m < 0.30:
            # Gặp vật cản trực diện: Chốt hướng quay dứt khoát trong 1.0 giây
            self.turn_direction = -1.0 if np.random.rand() > 0.5 else 1.0
            self.turn_timer = 1.0
            lin_vel = 0.0
            ang_vel = self.turn_direction * 0.75
        elif front_tof_m < 0.60:
            # Vật cản phía trước: Giảm tốc và bẻ lái nhẹ
            lin_vel = 0.12 * ((front_tof_m - 0.25) / 0.35)
            ang_vel = self.turn_direction * 0.45
        else:
            # Thông thoáng: Tiến nhanh hơn dựa trên mức độ tò mò/buồn chán
            base_speed = 0.20 + 0.10 * curiosity
            lin_vel = float(np.clip(base_speed, 0.15, 0.35))
            # Lượn nhẹ để quét môi trường
            ang_vel = float(0.20 * np.sin(self.turn_timer * 1.5))

        return np.array([lin_vel, ang_vel, pan_vel, tilt_vel, attn], dtype=np.float32)
