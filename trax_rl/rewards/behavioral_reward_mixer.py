import numpy as np
from typing import Dict, Any, Tuple

class BehavioralRewardMixer:
    """
    Composite Behavioral Reward Mixer (Doc 04 - Reward Design).
    Tổng hòa 11 thành phần thưởng/phạt và bộ lọc chống Reward Hacking:
      R = + w_nov * novelty
          + w_view * new_view
          + w_human * human_interest
          + w_eng * engagement
          + w_quiet * appropriate_quiet
          - w_col * collision
          - w_near * near_obstacle
          - w_jerk * motion_jerk
          - w_rep * repetition
          - w_spam * signal_spam
          - w_nrg * energy_cost
    """
    def __init__(self,
                 w_novelty: float = 0.25,
                 w_new_view: float = 0.15,
                 w_human: float = 0.20,
                 w_engagement: float = 0.25,
                 w_quiet: float = 0.45,
                 w_collision: float = 2.00,
                 w_near_obstacle: float = 0.30,
                 w_jerk: float = 0.20,
                 w_repetition: float = 0.15,
                 w_signal_spam: float = 0.50,
                 w_energy_cost: float = 0.15,
                 w_limit: float = 0.35,
                 w_override: float = 0.50,
                 w_stare: float = 1.20,
                 w_scan: float = 0.35,
                 w_excessive_movement: float = 0.60,
                 w_approach: float = 0.40,
                 w_discovery: float = 0.50,
                 w_dist_reduction: float = 0.50,
                 w_fixation: float = 0.35,
                 w_inspection: float = 0.40,
                 w_social_zone: float = 0.25,
                 w_crowding: float = 0.35,
                 w_reverse: float = 0.50,
                 w_switch: float = 0.25,
                 w_oscillation: float = 0.20):
        self.w_novelty = w_novelty
        self.w_new_view = w_new_view
        self.w_human = w_human
        self.w_engagement = w_engagement
        self.w_quiet = w_quiet
        self.w_collision = w_collision
        self.w_near_obstacle = w_near_obstacle
        self.w_jerk = w_jerk
        self.w_repetition = w_repetition
        self.w_signal_spam = w_signal_spam
        self.w_energy_cost = w_energy_cost
        self.w_limit = w_limit
        self.w_override = w_override
        self.w_stare = w_stare
        self.w_scan = w_scan
        self.w_excessive_movement = w_excessive_movement
        self.w_approach = w_approach
        self.w_discovery = w_discovery
        self.w_dist_reduction = w_dist_reduction
        self.w_fixation = w_fixation
        self.w_inspection = w_inspection
        self.w_social_zone = w_social_zone
        self.w_crowding = w_crowding
        self.w_reverse = w_reverse
        self.w_switch = w_switch
        self.w_oscillation = w_oscillation

        self.last_action = np.zeros(5, dtype=np.float32)
        self.last_velocity = np.zeros(4, dtype=np.float32)
        self.last_target_dist: float = 2.5
        self.last_target_visible: bool = False
        self.last_pan_sign: float = 0.0
        self.head_oscillation_count: int = 0

    def reset(self):
        self.last_action = np.zeros(5, dtype=np.float32)
        self.last_velocity = np.zeros(4, dtype=np.float32)
        self.last_target_dist = 2.5
        self.last_target_visible = False
        self.last_pan_sign = 0.0
        self.head_oscillation_count = 0

    def compute(self,
                action: np.ndarray,
                novelty_score: float,
                new_view_score: float,
                human_visible: bool,
                human_aligned: bool,
                human_engagement: float,
                front_obstacle_m: float,
                is_colliding: bool,
                is_spamming: bool,
                pan_norm: float = 0.0,
                tilt_norm: float = 0.0,
                is_overridden: bool = False,
                target_visible: bool = False,
                target_dist: float = 2.0,
                target_bearing: float = 0.0,
                gaze_duration: float = 0.0,
                pan_vel: float = 0.0,
                is_behavior_switched: bool = False,
                target_is_human: bool = True,
                dt: float = 0.1) -> Tuple[float, Dict[str, float]]:
        """
        Tính toán phần thưởng tổng hợp và phân rã các thành phần chi tiết (Doc 04, Doc 13).
        """
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 0:
            # Discrete scalar action
            energy_cost = 0.02 if int(action) == 0 else 0.20
            jerk = 0.0
            repetition = 1.0 if (int(action) == int(self.last_action[0]) and int(action) != 0) else 0.0
            current_act_vec = np.array([float(action), 0, 0, 0, 0], dtype=np.float32)
        else:
            # Continuous action [lin_vel, ang_vel, pan_vel, tilt_vel, attn_intensity]
            current_act_vec = action.copy()
            energy_cost = float(np.sum(np.abs(action[:4])) * 0.25 + action[4] * 0.10)

            # Tính Jerk (đạo hàm bậc 2 chuyển động để phạt giật cục)
            current_vel = action[:4]
            accel = current_vel - self.last_velocity
            jerk = float(np.sum(np.square(accel)))
            self.last_velocity = current_vel.copy()

            # Phạt lặp lại cùng hành vi vận động liên tục (tránh quay vòng vô tận)
            diff = float(np.linalg.norm(action - self.last_action))
            repetition = 1.0 if diff < 0.05 and energy_cost > 0.05 else 0.0

        self.last_action = current_act_vec.copy()

        # 1. Novelty reward: Thưởng cho thông tin mới
        r_novelty = self.w_novelty * novelty_score

        # 2. New view reward: Thưởng cho khám phá góc nhìn mới lạ
        r_new_view = self.w_new_view * new_view_score

        # 3. Human & Target interest reward with gaze decay (Doc 06, Doc 13 Section 7 & 10)
        r_human = 0.0
        has_target = human_visible or target_visible
        if has_target:
            gaze_discount = max(0.0, 1.0 - max(0.0, gaze_duration - 1.8) / 1.4)
            align_factor = 1.0 if human_aligned else 0.4
            r_human = self.w_human * align_factor * gaze_discount

        # 4. Engagement reward: Thưởng khi con người phản hồi tương tác
        r_engagement = self.w_engagement * human_engagement

        # 5. Appropriate quiet reward (Doc 04, Doc 13 Section 5):
        # Khi môi trường không có kích thích (not has_target) -> đứng yên tĩnh lặng và giữ đầu gần cân bằng
        # Target: 35-55% trong empty_room
        r_quiet = 0.0
        lin_v = float(action[0]) if (hasattr(action, "__len__") and len(action) > 0) else 0.0
        ang_v = float(action[1]) if (hasattr(action, "__len__") and len(action) > 1) else 0.0
        pan_v = float(action[2]) if (hasattr(action, "__len__") and len(action) > 2) else 0.0
        tilt_v = float(action[3]) if (hasattr(action, "__len__") and len(action) > 3) else 0.0
        base_speed = abs(lin_v) + abs(ang_v)
        head_speed = abs(pan_v) + abs(tilt_v)

        if not has_target:
            if base_speed < 0.08 and head_speed < 0.08:
                center_bonus = max(0.0, 1.0 - (pan_norm**2 + tilt_norm**2))
                r_quiet = self.w_quiet * (0.75 + 0.25 * center_bonus)

        # 6. View Scan & Exploration bonus (Doc 13 Section 6.4)
        # Chỉ thưởng scan khi có độ mới cao hoặc tái tìm kiếm mục tiêu; KHÔNG thưởng scan tự do trong phòng trống
        r_scan = 0.0
        if (novelty_score > 0.20 and not has_target) or (has_target and gaze_duration > 1.8):
            r_scan = self.w_scan * abs(pan_vel) * max(0.0, 1.0 - abs(pan_norm))

        # 7. Collision penalty
        p_collision = self.w_collision if is_colliding else 0.0

        # 8. Near obstacle penalty
        p_near_obstacle = 0.0
        if front_obstacle_m > 0 and front_obstacle_m < 0.25:
            p_near_obstacle = self.w_near_obstacle * (1.0 - front_obstacle_m / 0.25)

        # 9. Jerk penalty (Chống rung giật servo & bánh xích)
        p_jerk = self.w_jerk * np.clip(jerk, 0.0, 2.0)

        # 10. Repetition penalty (Chống lắc lư đầu vô tận)
        p_repetition = self.w_repetition * repetition

        # 11. Signal spam penalty (Phạt cố tình spam âm thanh / biểu cảm)
        p_spam = self.w_signal_spam if is_spamming else 0.0

        # 12. Energy cost penalty
        p_energy = self.w_energy_cost * energy_cost

        # 13. Servo Limit Saturation penalty (Doc 13 Section 5.4, 6.4)
        pan_excess = max(0.0, abs(pan_norm) - 0.80) / 0.20
        tilt_excess = max(0.0, abs(tilt_norm) - 0.80) / 0.20
        p_limit = self.w_limit * (pan_excess**2 + tilt_excess**2)

        # 14. Safety Override penalty (Doc 13 Section 5.4, 12.4)
        p_override = self.w_override if is_overridden else 0.0

        # 15. Anti-stare penalty (Doc 13 Section 7.4 & 10.4): Phạt khi nhìn chằm chằm > 2.5s
        p_stare = 0.0
        if gaze_duration > 2.5:
            p_stare = self.w_stare * min(1.5, (gaze_duration - 2.5) / 1.0)

        # 16. Excessive movement penalty in empty/idle setting (Doc 13 Section 5.4)
        p_excessive = 0.0
        if not has_target:
            if base_speed > 0.05 or head_speed > 0.08:
                p_excessive = self.w_excessive_movement * min(2.5, base_speed * 3.0 + head_speed * 1.5)

        # 17. Target approach & Object exploration rewards (Doc 06, Doc 13)
        r_approach = 0.0
        r_discovery = 0.0
        r_dist_reduction = 0.0
        r_fixation = 0.0
        r_inspection = 0.0
        r_social_zone = 0.0
        p_crowding = 0.0
        p_reverse = 0.0

        if target_visible:
            # A. Discovery reward: khi lần đầu phát hiện mục tiêu
            if not self.last_target_visible:
                r_discovery = self.w_discovery

            # B. Distance reduction (CAUSAL): chỉ thưởng phần tiến độ do CHÍNH robot tạo ra
            # (hình chiếu vận tốc tiến lên hướng mục tiêu). Trước đây dùng delta khoảng cách → 84% reward
            # đến từ chuyển động của bóng, PPO không gán được credit cho hành động (RC2).
            if self.last_target_visible and target_dist > 0.45:
                progress = lin_v * float(np.cos(target_bearing)) * dt
                if progress > 0.003:
                    r_dist_reduction = self.w_dist_reduction * min(1.0, progress * 12.0)

            # C. Stable fixation: Thưởng khi mục tiêu nằm ở trung tâm tầm nhìn (±12 độ)
            if abs(target_bearing) < 0.20:
                r_fixation = self.w_fixation * (1.0 - abs(target_bearing) / 0.20)

            # D. Inspection bonus: Thưởng khi duy trì cự ly quan sát tối ưu (0.35m - 0.65m)
            # Không thưởng khi đang lùi: chặn chiến lược "lùi trước bóng lao tới để giữ bóng trong dải" (RC2b).
            if 0.35 <= target_dist <= 0.65 and lin_v >= -0.01:
                r_inspection = self.w_inspection * (1.0 - abs(target_dist - 0.50) / 0.25)

            # E. Velocity-based approach:
            bearing_align = max(0.0, float(np.cos(target_bearing)))
            if target_dist > 0.70:
                r_approach = self.w_approach * float(np.clip(lin_v, 0.0, 0.4)) * bearing_align
            elif 0.45 <= target_dist <= 0.70:
                r_social_zone = self.w_social_zone * max(0.0, 1.0 - 2.5 * abs(lin_v))
            elif target_dist < 0.35:
                p_crowding = self.w_crowding * max(0.0, lin_v)

        # 18. Unprovoked Blind Reverse Penalty
        # Với người: miễn phạt trong vùng cá nhân (< 0.55m). Với đồ vật/bóng: lùi luôn bị phạt
        # (trừ khi phía trước có vật cản thật sự) vì robot không có cảm biến phía sau.
        if lin_v < -0.01 and front_obstacle_m >= 0.25 and (target_dist >= 0.55 or not target_is_human):
            p_reverse = self.w_reverse * abs(lin_v)

        # 19. Head Oscillation Penalty (chống lắc đầu lặp đi lặp lại)
        p_oscillation = 0.0
        current_pan_sign = np.sign(pan_v) if abs(pan_v) > 0.10 else 0.0
        if current_pan_sign != 0.0 and self.last_pan_sign != 0.0 and current_pan_sign != self.last_pan_sign:
            p_oscillation = self.w_oscillation
            self.head_oscillation_count += 1
        if current_pan_sign != 0.0:
            self.last_pan_sign = current_pan_sign

        # 20. Behavior Switching Penalty (phạt chuyển đổi hành vi quá nhanh)
        p_switch = self.w_switch if is_behavior_switched else 0.0

        # Cập nhật cache
        self.last_target_dist = float(target_dist)
        self.last_target_visible = bool(target_visible)

        # Tổng hợp Reward
        total_reward = (
            r_novelty
            + r_new_view
            + r_human
            + r_engagement
            + r_quiet
            + r_scan
            + r_approach
            + r_discovery
            + r_dist_reduction
            + r_fixation
            + r_inspection
            + r_social_zone
            - p_collision
            - p_near_obstacle
            - p_jerk
            - p_repetition
            - p_spam
            - p_energy
            - p_limit
            - p_override
            - p_stare
            - p_excessive
            - p_crowding
            - p_reverse
            - p_oscillation
            - p_switch
        )

        breakdown = {
            "reward/novelty": float(r_novelty),
            "reward/new_view": float(r_new_view),
            "reward/human": float(r_human),
            "reward/engagement": float(r_engagement),
            "reward/quiet": float(r_quiet),
            "reward/scan": float(r_scan),
            "reward/approach": float(r_approach),
            "reward/discovery": float(r_discovery),
            "reward/dist_reduction": float(r_dist_reduction),
            "reward/fixation": float(r_fixation),
            "reward/inspection": float(r_inspection),
            "reward/social_zone": float(r_social_zone),
            "penalty/collision": float(p_collision),
            "penalty/near_obstacle": float(p_near_obstacle),
            "penalty/jerk": float(p_jerk),
            "penalty/repetition": float(p_repetition),
            "penalty/signal_spam": float(p_spam),
            "penalty/energy": float(p_energy),
            "penalty/servo_limit": float(p_limit),
            "penalty/safety_override": float(p_override),
            "penalty/stare": float(p_stare),
            "penalty/excessive_movement": float(p_excessive),
            "penalty/crowding": float(p_crowding),
            "penalty/reverse": float(p_reverse),
            "penalty/head_oscillation": float(p_oscillation),
            "penalty/behavior_switch": float(p_switch),
            "reward/total": float(total_reward)
        }

        return float(total_reward), breakdown
