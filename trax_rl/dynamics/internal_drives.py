import numpy as np

class InternalDrivesEngine:
    """
    T-RAX Internal Drives Engine (Doc 01, Doc 03, Doc 11 & Upgrade Plan Section 3.1 & 12).
    Quản lý các động lực nội tại (Homeostatic Drives):
      1. curiosity:           Độ tò mò muốn khám phá cái mới [0.0, 1.0]
      2. boredom:             Độ buồn chán do môi trường tĩnh/hành động lặp [0.0, 1.0]
      3. social_need:         Nhu cầu tương tác xã hội [0.0, 1.0]
      4. energy_budget:       Ngân sách năng lượng còn lại [0.0, 1.0]
      5. perception_confidence: Độ tin cậy cảm biến/thị giác [0.0, 1.0]
      6. fear (threat drive): Động lực sợ hãi/nguy cơ kích hoạt khi phát hiện đe dọa [0.0, 1.0]
      7. arousal:             Mức độ kích hoạt sinh lý/thần kinh dựa trên threat & novelty [0.0, 1.0]
      8. habituation:         Mức độ quen dần với kích thích lặp lại [0.0, 1.0]
    """
    def __init__(self,
                 curiosity_decay: float = 0.05,
                 curiosity_growth: float = 0.02,
                 boredom_growth: float = 0.03,
                 social_growth: float = 0.015,
                 energy_recharge_rate: float = 0.01,
                 energy_burn_rate: float = 0.005,
                 threat_gain: float = 3.0,
                 fear_decay: float = 0.8,
                 arousal_baseline: float = 0.20,
                 arousal_threat_gain: float = 2.0,
                 arousal_novelty_gain: float = 0.8,
                 arousal_decay: float = 0.5):
        self.curiosity_decay = curiosity_decay
        self.curiosity_growth = curiosity_growth
        self.boredom_growth = boredom_growth
        self.social_growth = social_growth
        self.energy_recharge_rate = energy_recharge_rate
        self.energy_burn_rate = energy_burn_rate

        self.threat_gain = threat_gain
        self.fear_decay = fear_decay
        self.arousal_baseline = arousal_baseline
        self.arousal_threat_gain = arousal_threat_gain
        self.arousal_novelty_gain = arousal_novelty_gain
        self.arousal_decay = arousal_decay

        self.reset()

    def reset(self, initial_energy: float = 1.0):
        self.curiosity = 0.5
        self.boredom = 0.1
        self.social_need = 0.3
        self.energy_budget = float(initial_energy)
        self.readiness = 1.0
        self.perception_confidence = 0.95
        self.fear = 0.0
        self.arousal = self.arousal_baseline
        self.habituation = 0.0

    def step(self,
             dt: float,
             novelty_score: float,
             human_visible: bool,
             human_engaged: bool,
             action_energy: float,
             is_resting: bool,
             threat_score: float = 0.0,
             sensor_loss: bool = False,
             charging: bool = False):
        """
        Cập nhật trạng thái động lực nội tại qua từng bước thời gian dt (10Hz).
        Động lực là Dynamic State, không phải timer tăng tuyến tính:
        - Khi có threat (vật cản quá gần, âm thanh bất ngờ, va đập): fear & arousal tăng nhanh.
        - Khi môi trường an toàn (threat = 0): fear tự phân rã (exponential decay) về 0.
        """
        # 1. Cập nhật Curiosity:
        # Nếu gặp cái mới (novelty cao) -> giải tỏa tò mò (drop), nếu lâu không có gì mới -> tích lũy tò mò
        if novelty_score > 0.4:
            self.curiosity = max(0.1, self.curiosity - self.curiosity_decay * novelty_score * dt)
        else:
            self.curiosity = min(1.0, self.curiosity + self.curiosity_growth * dt)

        # 2. Cập nhật Boredom:
        # Nếu môi trường không đổi và hành vi đơn điệu -> tăng độ buồn chán; nếu có biến cố/người -> giảm về 0
        if human_engaged or novelty_score > 0.3:
            self.boredom = max(0.0, self.boredom - 0.2 * dt)
        else:
            self.boredom = min(1.0, self.boredom + self.boredom_growth * dt)

        # 3. Cập nhật Social Need:
        # Tăng dần khi ở một mình, giảm mạnh khi được con người chú ý/tương tác
        if human_engaged:
            self.social_need = max(0.05, self.social_need - 0.3 * dt)
        else:
            self.social_need = min(1.0, self.social_need + self.social_growth * dt)

        # 4. Cập nhật Energy Budget:
        # Tiêu hao dựa trên cường độ cử động, hồi phục chậm khi robot nghỉ ngơi (REST)
        if charging:
            self.energy_budget = min(1.0, self.energy_budget + self.energy_recharge_rate * dt)
        else:
            burn = self.energy_burn_rate * (0.1 + action_energy) * dt
            self.energy_budget = max(0.0, self.energy_budget - burn)
        self.readiness = float(np.clip(self.readiness +
            (0.04 if is_resting else -0.01 * action_energy) * dt, 0, 1))

        # 5. Cập nhật Perception Confidence:
        if sensor_loss:
            self.perception_confidence = max(0.1, self.perception_confidence - 0.3 * dt)
        else:
            self.perception_confidence = min(1.0, self.perception_confidence + 0.1 * dt)

        # 6. Cập nhật Fear (Threat / Safety Drive) - Upgrade Plan Section 3.1:
        # threat detected -> fear up; safe environment -> fear decay
        self.fear = float(np.clip(self.fear + (self.threat_gain * threat_score - self.fear_decay * self.fear) * dt, 0.0, 1.0))

        # 7. Cập nhật Arousal (Mức kích hoạt sinh lý) - Upgrade Plan Section 12:
        arousal_delta = (
            self.arousal_threat_gain * threat_score
            + self.arousal_novelty_gain * novelty_score
            - self.arousal_decay * (self.arousal - self.arousal_baseline)
        ) * dt
        self.arousal = float(np.clip(self.arousal + arousal_delta, 0.0, 1.0))

        # 8. Cập nhật Habituation (Quen dần với kích thích lặp lại):
        if novelty_score > 0.2:
            self.habituation = min(1.0, self.habituation + 0.05 * dt)
        else:
            self.habituation = max(0.0, self.habituation - 0.02 * dt)

    def get_vector(self) -> np.ndarray:
        """
        Trả về vector 5 chiều chuẩn hóa [0.0 .. 1.0] tương thích với Observation Schema v1
        """
        return np.array([
            self.curiosity,
            self.boredom,
            self.social_need,
            self.energy_budget,
            self.perception_confidence
        ], dtype=np.float32)

    def get_extended_vector(self) -> np.ndarray:
        """
        Trả về vector 7 chiều đầy đủ [curiosity, boredom, social_need, energy_budget, confidence, fear, arousal]
        """
        return np.array([
            self.curiosity,
            self.boredom,
            self.social_need,
            self.energy_budget,
            self.perception_confidence,
            self.fear,
            self.arousal
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "curiosity": float(self.curiosity),
            "boredom": float(self.boredom),
            "social_need": float(self.social_need),
            "energy_budget": float(self.energy_budget),
            "readiness": float(self.readiness),
            "perception_confidence": float(self.perception_confidence),
            "fear": float(self.fear),
            "safety": float(1.0 - self.fear),
            "arousal": float(self.arousal),
            "habituation": float(self.habituation)
        }
