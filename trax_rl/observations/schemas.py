import numpy as np

def build_idle_observation(target_visible: bool,
                           target_x_norm: float,
                           motion_level: float,
                           head_pan_rad: float,
                           max_pan_rad: float,
                           time_since_event_sec: float,
                           prev_action_id: int) -> np.ndarray:
    """
    Chuẩn hóa Observation Vector 6 chiều khớp 100% với WorldState trên Firmware ESP-IDF:
    Index 0: target_visible       [0.0, 1.0]
    Index 1: target_x_norm        [-1.0, 1.0]
    Index 2: motion_level         [0.0, 1.0]
    Index 3: head_pan_norm        [-1.0, 1.0]
    Index 4: time_since_event     [0.0, 1.0] (clipped at 10.0s)
    Index 5: prev_action_norm     [-1.0, 1.0]
    """
    vis = 1.0 if target_visible else 0.0
    x_n = np.clip(target_x_norm, -1.0, 1.0)
    mot = np.clip(motion_level, 0.0, 1.0)
    pan_n = np.clip(head_pan_rad / max_pan_rad, -1.0, 1.0)
    t_event_n = np.clip(time_since_event_sec / 10.0, 0.0, 1.0)
    action_n = (float(prev_action_id) - 2.0) / 2.0  # Mapping [0..4] sang [-1.0 .. 1.0]

    obs = np.array([vis, x_n, mot, pan_n, t_event_n, action_n], dtype=np.float32)
    return obs
