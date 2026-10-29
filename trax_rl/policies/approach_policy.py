import numpy as np
from typing import Dict, Any, Optional
from trax_rl.policies.base_policy import BasePolicy

class ApproachPolicy(BasePolicy):
    """
    Approach Policy (Upgrade Plan Phase 5):
    Chuyên gia bám theo và tiến lại gần đối tượng mục tiêu (Ball, Novel Object, Human)
    sử dụng thị giác kết hợp điều hòa khoảng cách từ DUY NHẤT 1 CẢM BIẾN ToF.
    
    Quy tắc an toàn & tương tác:
    - Xoay hướng đế xích bám theo góc lệch mục tiêu (Visual Bearing).
    - Giảm tốc độ tuyến tính mượt mà khi tiến sát.
    - Dừng hẳn tại khoảng cách an toàn (Social / Inspection distance 0.55m - 0.65m)
      để tránh xâm lấn không gian hoặc va quẹt.
    """
    def __init__(self, name: str = "approach_policy", target_stop_dist: float = 0.60):
        super().__init__(name)
        self.target_stop_dist = target_stop_dist

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # Trích xuất góc lệch mục tiêu từ Vision observation
        # obs[9] = face_x hoặc obs[14] = object_x trong 39-dim schema
        face_vis = bool(obs[8] > 0.5) if len(obs) > 8 else False
        obj_vis = bool(obs[13] > 0.5) if len(obs) > 13 else False
        
        target_x = float(obs[9]) if face_vis else (float(obs[14]) if obj_vis else 0.0)
        target_y = float(obs[10]) if face_vis else (float(obs[15]) if obj_vis else 0.0)

        # Trích xuất cự ly đo từ 1 cảm biến ToF phía trước
        front_tof_m = float(obs[6]) * 3.0 if len(obs) > 6 else 2.0

        # Quay đế xe và đầu để hướng thẳng vào mục tiêu
        ang_vel = float(np.clip(-target_x * 1.5, -0.8, 0.8))
        pan_vel = float(np.clip(-target_x * 2.0, -1.0, 1.0))
        tilt_vel = float(np.clip(-target_y * 1.5, -0.8, 0.8))

        # Điều hòa vận tốc tiến dựa trên cự ly ToF
        dist_err = front_tof_m - self.target_stop_dist
        if dist_err > 0.10:
            lin_vel = float(np.clip(dist_err * 0.40, 0.0, 0.30))
        else:
            lin_vel = 0.0  # Đã đạt cự ly quan sát tối ưu

        attn = 0.50 if face_vis else 0.35

        return np.array([lin_vel, ang_vel, pan_vel, tilt_vel, attn], dtype=np.float32)
