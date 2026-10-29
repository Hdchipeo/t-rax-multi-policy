import numpy as np
from typing import List, Tuple

class EpisodicNoveltyEngine:
    """
    Episodic Novelty & Curiosity Engine (Doc 05 - Curiosity & Novelty).
    Nguyên tắc: Thưởng cho Information Gain / Novelty, chống reward-hacking do quay đầu vô tận.
    - Duy trì episodic memory M = {z_1, z_2, ..., z_n}
    - Tính khoảng cách L2 tối thiểu tới các trạng thái đã biết
    - Áp dụng Novelty Decay khi nhìn chăm chăm vào một điểm (Anti-stare)
    """
    def __init__(self,
                 feature_dim: int = 6,
                 max_memory_size: int = 1024,
                 novelty_threshold: float = 0.25,
                 decay_rate: float = 0.92):
        self.feature_dim = feature_dim
        self.max_memory_size = max_memory_size
        self.novelty_threshold = novelty_threshold
        self.decay_rate = decay_rate

        self.memory: List[np.ndarray] = []
        self.gaze_dwell_time: float = 0.0
        self.last_compact_state: np.ndarray = np.zeros(feature_dim, dtype=np.float32)

        # Telemetry metrics
        self.novel_states_count: int = 0
        self.total_novelty_accum: float = 0.0
        self.steps_evaluated: int = 0
        self.repeat_visit_count: int = 0

    def reset(self):
        """Khởi tạo lại bộ nhớ khi bắt đầu một episode mới"""
        self.memory.clear()
        self.gaze_dwell_time = 0.0
        self.last_compact_state = np.zeros(self.feature_dim, dtype=np.float32)
        self.novel_states_count = 0
        self.total_novelty_accum = 0.0
        self.steps_evaluated = 0
        self.repeat_visit_count = 0

    def compute_compact_state(self,
                              robot_pan_norm: float,
                              robot_tilt_norm: float,
                              target_visible: bool,
                              target_bearing_norm: float,
                              target_distance_norm: float,
                              motion_score: float) -> np.ndarray:
        """
        Chiếu trạng thái quan sát thành vector đặc trưng compact z 6 chiều:
        [pan, tilt, target_vis, bearing, dist, motion]
        """
        vis_val = 1.0 if target_visible else 0.0
        z = np.array([
            float(np.clip(robot_pan_norm, -1.0, 1.0)),
            float(np.clip(robot_tilt_norm, -1.0, 1.0)),
            vis_val,
            float(np.clip(target_bearing_norm, -1.0, 1.0)),
            float(np.clip(target_distance_norm, 0.0, 1.0)),
            float(np.clip(motion_score, 0.0, 1.0))
        ], dtype=np.float32)
        return z

    def evaluate(self, compact_state: np.ndarray, dt: float = 0.1) -> Tuple[float, float, float]:
        """
        Đánh giá độ mới (Novelty Score) của trạng thái hiện tại.
        Returns:
            novelty_score:           Điểm mới lạ tức thời sau khi tính decay [0.0 .. 1.0]
            current_view_novelty:    Độ mới so với góc nhìn trước đó
            object_novelty:          Độ mới liên quan tới vật thể quan sát được
        """
        self.steps_evaluated += 1

        # 1. Đo độ lệch so với góc nhìn trước đó (View delta)
        delta_last = float(np.linalg.norm(compact_state - self.last_compact_state))
        if delta_last < 0.08:
            # Robot đang nhìn chăm chăm vào cùng 1 điểm -> tăng dwell time
            self.gaze_dwell_time += dt
        else:
            # Robot đã chuyển hướng nhìn -> reset dwell time
            self.gaze_dwell_time = 0.0

        self.last_compact_state = compact_state.copy()

        # 2. Tính khoảng cách tối thiểu tới toàn bộ Episodic Memory
        if len(self.memory) == 0:
            min_dist = 1.0
            raw_novelty = 1.0
            self.memory.append(compact_state)
            self.novel_states_count += 1
        else:
            # Vectorized nearest neighbor search
            mem_array = np.vstack(self.memory)
            dists = np.linalg.norm(mem_array - compact_state, axis=1)
            min_dist = float(np.min(dists))

            # Raw novelty chuẩn hóa qua hàm bão hòa tanh
            raw_novelty = float(np.tanh(min_dist / 0.5))

            # Lưu vào bộ nhớ nếu vượt ngưỡng độ mới và chưa đầy
            if min_dist > self.novelty_threshold and len(self.memory) < self.max_memory_size:
                self.memory.append(compact_state)
                self.novel_states_count += 1
            elif min_dist <= 0.10:
                self.repeat_visit_count += 1

        # 3. Áp dụng Gaze Dwell Decay (Anti-stare penalty - Doc 04 & 05)
        # Nếu chăm chú nhìn một vật quá lâu, novelty giảm dần theo hàm mũ
        decay_factor = float(self.decay_rate ** (self.gaze_dwell_time / 0.5))
        novelty_score = float(np.clip(raw_novelty * decay_factor, 0.0, 1.0))

        # 4. Tách các thành phần thăm dò (Exploration vector 4-dim cho Observation):
        current_view_novelty = float(np.clip(delta_last / 0.5, 0.0, 1.0))
        object_novelty = novelty_score if compact_state[2] > 0.5 else 0.0

        self.total_novelty_accum += novelty_score

        return novelty_score, current_view_novelty, object_novelty

    def get_metrics(self) -> dict:
        mean_nov = self.total_novelty_accum / max(1, self.steps_evaluated)
        return {
            "mean_novelty": float(mean_nov),
            "novel_states_per_episode": int(self.novel_states_count),
            "memory_size": int(len(self.memory)),
            "repeat_visit_count": int(self.repeat_visit_count),
            "gaze_dwell_time": float(self.gaze_dwell_time)
        }
