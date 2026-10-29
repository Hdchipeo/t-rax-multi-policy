import numpy as np
from typing import Tuple, Optional

class HumanAgent:
    """
    Simulated Human Social Agent (Doc 06 - Social Attention).
    Mô phỏng phản ứng của con người đối với hành vi của robot:
    - Đáp ứng khi robot nhìn vào mình một cách tự nhiên
    - Thích thú khi robot nghiêng đầu / gật đầu / phát tín hiệu vừa phải
    - Khó chịu, làm ngơ khi robot spam tín hiệu âm thanh / chuyển động giật cục
    """
    def __init__(self,
                 base_reaction_prob: float = 0.05,
                 gaze_sensitivity: float = 0.30,
                 gesture_sensitivity: float = 0.35,
                 audio_sensitivity: float = 0.40,
                 spam_penalty: float = 0.45):
        self.base_reaction_prob = base_reaction_prob
        self.gaze_sensitivity = gaze_sensitivity
        self.gesture_sensitivity = gesture_sensitivity
        self.audio_sensitivity = audio_sensitivity
        self.spam_penalty = spam_penalty
        self.rng = np.random.default_rng()

        self.is_present: bool = True
        self.engagement_level: float = 0.0   # [0.0 .. 1.0] Mức độ gắn kết tương tác
        self.attention_span_timer: float = 0.0
        self.position: np.ndarray = np.array([0.0, 1.2, 0.0], dtype=np.float32)

    def reset(self, is_present: bool = True, custom_pos: Optional[np.ndarray] = None):
        self.is_present = is_present
        self.engagement_level = 0.0
        self.attention_span_timer = 0.0
        if custom_pos is not None:
            self.position = custom_pos.copy()
            self.position[2] = 0.0  # Floor-level base
        else:
            self.position = np.array([0.0, 1.2, 0.0], dtype=np.float32)

    def update(self,
               dt: float,
               robot_looking_at_human: bool,
               robot_head_gesture: bool,
               audio_signal_emitted: bool,
               signal_spam: bool) -> bool:
        """
        Cập nhật phản ứng của con người. Trả về True nếu con người phản hồi tương tác (gật đầu, vẫy tay).
        """
        if not self.is_present:
            self.engagement_level = 0.0
            return False

        # Tính xác suất phản hồi tức thời
        p = self.base_reaction_prob
        if robot_looking_at_human:
            p += self.gaze_sensitivity
        if robot_head_gesture:
            p += self.gesture_sensitivity
        if audio_signal_emitted:
            p += self.audio_sensitivity
        if signal_spam:
            p -= self.spam_penalty

        p_clipped = float(np.clip(p, 0.0, 0.95))
        # Parameters are probabilities per 0.1s reference interval.
        has_responded = (self.rng.random() < 1.0 - (1.0 - p_clipped) ** (dt / .1))

        # Cập nhật mức độ gắn kết tương tác (Engagement Level)
        if has_responded:
            self.engagement_level = min(1.0, self.engagement_level + 0.25)
            self.attention_span_timer = 3.0  # Giữ tương tác trong 3 giây
        else:
            if self.attention_span_timer > 0:
                self.attention_span_timer -= dt
                self.engagement_level = max(0.0, self.engagement_level - 0.05 * dt)
            else:
                self.engagement_level = max(0.0, self.engagement_level - 0.15 * dt)

        return has_responded


class SocialAttentionDynamics:
    """
    Quản lý luồng tương tác xã hội & giải mã tín hiệu Attention (Doc 06 & Doc 07).
    - Giải mã attention_intensity [0.0 .. 1.0] thành các tầng biểu cảm
    - Kiểm soát cooldown âm thanh/tín hiệu (anti-spam)
    - Theo dõi thời gian chuỗi nhìn (gaze saturation / brief gaze / look away)
    - Xác định thời điểm bàn giao sang SocialPolicy (Policy Handoff)
    """
    def __init__(self,
                 audio_cooldown_sec: float = 3.0,
                 handoff_threshold: float = 0.75,
                 handoff_min_duration_sec: float = 2.0):
        self.audio_cooldown_sec = audio_cooldown_sec
        self.handoff_threshold = handoff_threshold
        self.handoff_min_duration_sec = handoff_min_duration_sec

        self.time_since_audio_signal: float = audio_cooldown_sec
        self.time_since_human_seen: float = 10.0
        self.time_since_human_response: float = 10.0
        self.continuous_gaze_duration: float = 0.0
        self.sustained_engagement_duration: float = 0.0

        # Expression tier outputs
        self.current_tier: int = 0
        self.is_spamming: bool = False

    def reset(self):
        self.time_since_audio_signal = self.audio_cooldown_sec
        self.time_since_human_seen = 10.0
        self.time_since_human_response = 10.0
        self.continuous_gaze_duration = 0.0
        self.sustained_engagement_duration = 0.0
        self.current_tier = 0
        self.is_spamming = False

    def decode_attention_intensity(self, intensity: float, dt: float) -> Tuple[int, bool, bool, bool]:
        """
        Giải mã attention_intensity ∈ [0.0, 1.0] theo 5 tầng biểu cảm:
          0.0 - 0.2: Tier 0 (Nothing/Calm)
          0.2 - 0.4: Tier 1 (Micro head motion)
          0.4 - 0.6: Tier 2 (Gaze lock + head gesture)
          0.6 - 0.8: Tier 3 (Stronger expressive alert gesture)
          0.8 - 1.0: Tier 4 (Audio R2D2 chirp / Eye LED strobe - subjected to cooldown)
        Returns:
            tier:               Tầng biểu cảm (0 .. 4)
            has_head_gesture:   Có cử động đầu thể hiện không
            emit_audio:         Có kích hoạt chirp âm thanh không
            is_spam:            Có vi phạm spam tín hiệu không
        """
        self.time_since_audio_signal += dt
        clamped_intensity = float(np.clip(intensity, 0.0, 1.0))

        if clamped_intensity < 0.20:
            tier = 0
        elif clamped_intensity < 0.40:
            tier = 1
        elif clamped_intensity < 0.60:
            tier = 2
        elif clamped_intensity < 0.80:
            tier = 3
        else:
            tier = 4

        self.current_tier = tier
        has_head_gesture = (tier in (2, 3))
        emit_audio = False
        is_spam = False

        if tier == 4:
            if self.time_since_audio_signal >= self.audio_cooldown_sec:
                emit_audio = True
                self.time_since_audio_signal = 0.0
                is_spam = False
            else:
                # Cố tình phát âm thanh khi chưa hết cooldown -> tính là spam
                emit_audio = False
                is_spam = True

        self.is_spamming = is_spam
        return tier, has_head_gesture, emit_audio, is_spam

    def step(self,
             dt: float,
             human_visible: bool,
             human_aligned: bool,
             human_responded: bool,
             engagement_level: float) -> bool:
        """
        Cập nhật trạng thái chuỗi thời gian xã hội.
        Returns:
            ready_for_social_handoff: True nếu đủ điều kiện chuyển từ IdlePolicy sang SocialPolicy
        """
        if human_visible:
            self.time_since_human_seen = 0.0
            if human_aligned:
                self.continuous_gaze_duration += dt
            else:
                self.continuous_gaze_duration = max(0.0, self.continuous_gaze_duration - dt * 0.5)
        else:
            self.time_since_human_seen += dt
            self.continuous_gaze_duration = 0.0

        if human_responded:
            self.time_since_human_response = 0.0
        else:
            self.time_since_human_response += dt

        # Kiểm tra điều kiện bàn giao sang SocialPolicy (Handoff Hysteresis - Doc 01, Doc 06)
        if engagement_level >= self.handoff_threshold and human_visible:
            self.sustained_engagement_duration += dt
        else:
            self.sustained_engagement_duration = max(0.0, self.sustained_engagement_duration - dt)

        ready_for_handoff = (self.sustained_engagement_duration >= self.handoff_min_duration_sec)
        return ready_for_handoff

    def get_social_state_dict(self) -> dict:
        return {
            "tier": self.current_tier,
            "is_spamming": self.is_spamming,
            "time_since_human_seen": float(self.time_since_human_seen),
            "time_since_human_response": float(self.time_since_human_response),
            "time_since_audio_signal": float(self.time_since_audio_signal),
            "continuous_gaze_duration": float(self.continuous_gaze_duration),
            "sustained_engagement_duration": float(self.sustained_engagement_duration)
        }
