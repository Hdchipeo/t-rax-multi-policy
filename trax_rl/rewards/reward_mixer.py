import numpy as np

class IdleRewardMixer:
    """
    Module hóa hàm tính Reward cho Autonomous Idle:
    - Thưởng khi bám đúng người/vật thể khi xuất hiện
    - Thưởng khi nghỉ ngơi (REST) khi môi trường không có sự kiện (tiết kiệm năng lượng)
    - Phạt hành vi lặp lại (tránh giật lag, co giật)
    - Phạt tiêu hao năng lượng khi chuyển động không cần thiết
    """
    def __init__(self,
                 w_attention: float = 0.5,
                 w_center: float = 0.3,
                 w_rest: float = 0.05,
                 w_repetition: float = -0.03,
                 w_energy: float = -0.01):
        self.w_attention = w_attention
        self.w_center = w_center
        self.w_rest = w_rest
        self.w_repetition = w_repetition
        self.w_energy = w_energy

    def compute(self, obs: np.ndarray, action: int, prev_action: int, energy_cost: float) -> float:
        target_visible = obs[0] > 0.5
        target_x_norm = obs[1]
        
        reward = 0.0

        if target_visible:
            # Khi thấy mục tiêu: khuyến khích TRACK_TARGET (action 4)
            if action == 4:
                # Độ lệch tâm càng nhỏ (càng hướng vào giữa) điểm càng cao
                alignment = max(0.0, 1.0 - abs(target_x_norm))
                reward += self.w_attention + self.w_center * alignment
            elif action in (1, 2, 3):
                # Nhìn ngó xung quanh khi mục tiêu ở xa tâm
                reward += 0.1
        else:
            # Khi không có mục tiêu: ưu tiên REST (action 0) để tĩnh lặng và tiết kiệm pin
            if action == 0:
                reward += self.w_rest
            elif action == 4:
                # Phạt nếu cố tình TRACK_TARGET khi không thấy ai
                reward -= 0.1

        # Phạt lặp lại cùng một hành vi liên tục (ngoại trừ REST)
        if action == prev_action and action != 0:
            reward += self.w_repetition

        # Phạt năng lượng tiêu thụ
        reward += self.w_energy * energy_cost

        return float(reward)
