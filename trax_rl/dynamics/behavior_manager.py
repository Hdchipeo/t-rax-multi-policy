import numpy as np
from enum import IntEnum
from typing import Tuple, Dict, Any, Optional, List

class Behavior(IntEnum):
    SAFE_STOP = 0
    COMMAND = 1
    RECOVERY = 2
    SOCIAL = 3
    TRACK = 4
    BALL = 5
    IDLE = 6
    EXPLORE = 7
    LOOK_AROUND = 8
    APPROACH = 9
    ESCAPE = 10

class BehaviorManager:
    """
    T-RAX Behavior Manager & Priority Arbiter (Doc 01, Doc 07 & Upgrade Plan Section 4 & 14).
    Chịu trách nhiệm:
    - Quản lý cây ưu tiên hành vi (Priority Hierarchy) & Multi-Policy Handoff
    - Cơ chế chống rung lật (Hysteresis): enter_threshold, exit_threshold, minimum_duration, cooldown
    - Lưu vết giải thích nguyên nhân kích hoạt (Trigger Reason) & Policy tương ứng
    - Lớp an toàn tiền thực thi (Deterministic Safety Layer)
    - So sánh Raw Action vs Safe Action và telemetry can thiệp an toàn
    """
    def __init__(self,
                 min_active_time_sec: float = 2.5,
                 cooldown_sec: float = 1.0,
                 explore_enter_threshold: float = 0.65,
                 explore_exit_threshold: float = 0.35):
        self.min_active_time_sec = min_active_time_sec
        self.cooldown_sec = cooldown_sec
        self.explore_enter_threshold = explore_enter_threshold
        self.explore_exit_threshold = explore_exit_threshold

        self.current_behavior: Behavior = Behavior.IDLE
        self.behavior_trigger: str = "INITIAL_IDLE"
        self.active_policy: str = "autonomous_idle_policy"
        self.active_time_sec: float = 0.0
        self.cooldown_timer_sec: float = 0.0

        # Safety override metrics & raw/safe actions
        self.total_commands_evaluated: int = 0
        self.total_overrides_triggered: int = 0
        self.last_override_reason: str = "NONE"
        self.last_raw_action: np.ndarray = np.zeros(5, dtype=np.float32)
        self.last_safe_action: np.ndarray = np.zeros(5, dtype=np.float32)
        self.last_raw_left: float = 0.0
        self.last_raw_right: float = 0.0
        self.last_safe_left: float = 0.0
        self.last_safe_right: float = 0.0
        self.last_intervention: str = "NONE"
        self.action_changed_count = 0
        self.correction_counts = {}
        self.last_corrections = []

        self.disengage_timer: float = 0.0
        self.disengage_direction: float = 0.0
        self.scan_direction: float = 0.70
        self.last_safe_lin: float = 0.0
        self.last_safe_ang: float = 0.0
        self.last_safe_pan_v: float = 0.0
        self.last_safe_tilt_v: float = 0.0
        self.avoid_steer_dir: float = 0.0
        self.avoid_steer_timer: float = 0.0
        self.stuck_counter: int = 0

        # Switch & Dwell time tracking
        self.switch_counter: int = 0
        self.behavior_durations: List[float] = []
        self.last_transition_switched: bool = False
        self.last_behavior: Behavior = Behavior.IDLE

        # Multi-Policy Architecture (Upgrade Plan Phase 3 & 4)
        from trax_rl.policies.explore_policy import ExplorePolicy
        from trax_rl.policies.look_around_policy import LookAroundPolicy
        from trax_rl.policies.approach_policy import ApproachPolicy
        from trax_rl.policies.threat_escape_policy import ThreatEscapePolicy

        shared_escape = ThreatEscapePolicy()
        self.policy_registry: Dict[Behavior, Any] = {
            Behavior.EXPLORE: ExplorePolicy(),
            Behavior.LOOK_AROUND: LookAroundPolicy(),
            Behavior.APPROACH: ApproachPolicy(),
            Behavior.TRACK: ApproachPolicy(name="track_policy"),
            Behavior.BALL: ApproachPolicy(name="ball_interact_policy", target_stop_dist=0.35),
            Behavior.ESCAPE: shared_escape,
            Behavior.RECOVERY: shared_escape
        }

    def register_policy(self, behavior: Behavior, policy: Any):
        """Đăng ký chuyên gia Policy (RL checkpoint hoặc heuristic) cho hành vi tương ứng"""
        self.policy_registry[behavior] = policy

    def get_action_from_policy(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Thực thi suy luận từ Policy tương ứng với hành vi được trọng tài lựa chọn"""
        policy = self.policy_registry.get(self.current_behavior)
        if policy is not None and hasattr(policy, "act"):
            return policy.act(obs, info)
        # Mặc định: hành vi IDLE tĩnh lặng cơ học
        return np.array([0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)

    def reset(self):
        self.current_behavior = Behavior.IDLE
        self.behavior_trigger = "INITIAL_IDLE"
        self.active_policy = "autonomous_idle_policy"
        self.active_time_sec = 0.0
        self.cooldown_timer_sec = 0.0
        self.total_commands_evaluated = 0
        self.total_overrides_triggered = 0
        self.last_override_reason = "NONE"
        self.last_raw_action = np.zeros(5, dtype=np.float32)
        self.last_safe_action = np.zeros(5, dtype=np.float32)
        self.last_raw_left = 0.0
        self.last_raw_right = 0.0
        self.last_safe_left = 0.0
        self.last_safe_right = 0.0
        self.last_intervention = "NONE"
        self.action_changed_count = 0
        self.correction_counts = {}
        self.last_corrections = []
        self.disengage_timer = 0.0
        self.disengage_direction = 0.0
        self.scan_direction = 0.70
        self.last_safe_lin = 0.0
        self.last_safe_ang = 0.0
        self.last_safe_pan_v = 0.0
        self.last_safe_tilt_v = 0.0
        self.avoid_steer_dir = 0.0
        self.avoid_steer_timer = 0.0
        self.stuck_counter = 0
        self.switch_counter = 0
        self.behavior_durations = []
        self.last_transition_switched = False
        for policy in set(self.policy_registry.values()):
            policy.reset()

    def arbitrate(self,
                  dt: float,
                  emergency_stop: bool = False,
                  user_command_active: bool = False,
                  obstacle_dist_m: float = 2.5,
                  human_engagement: float = 0.0,
                  ball_confidence: float = 0.0,
                  target_visible: bool = False,
                  curiosity: float = 0.5,
                  boredom: float = 0.1,
                  fear: float = 0.0,
                  threat_active: bool = False,
                  target_dist: float = 2.0,
                  social_need: float = 0.3) -> Behavior:
        """
        Trọng tài hành vi thời gian thực với Hysteresis & Cooldown.
        Priority Hierarchy (Upgrade Plan Section 4 & 7.7):
          Emergency > User Command > Critical Threat/Escape > Safety Recovery > Social/Ball Prioritization > Explore > Idle
        """
        self.active_time_sec += dt
        self.last_transition_switched = False
        if self.cooldown_timer_sec > 0:
            self.cooldown_timer_sec = max(0.0, self.cooldown_timer_sec - dt)

        # 1. Cấp ưu tiên tuyệt đối: Emergency Stop
        if emergency_stop:
            self._transition_to(Behavior.SAFE_STOP, trigger="EMERGENCY_STOP", policy="safe_stop")
            return self.current_behavior

        # 2. Cấp ưu tiên 2: User Command (Điều khiển thủ công từ Web/App/Voice)
        if user_command_active:
            self._transition_to(Behavior.COMMAND, trigger="USER_COMMAND", policy="teleop_command")
            return self.current_behavior

        # 3. Cấp ưu tiên 3: Critical Threat / Escape (Upgrade Plan Section 7.4)
        # Nếu đang trong quá trình RECOVERY né cản, không ngắt quãng sang ESCAPE trừ khi có nguy cơ đe dọa cực lớn (fear >= 0.70)
        if (threat_active and self.current_behavior != Behavior.RECOVERY) or fear >= 0.70:
            self._transition_to(Behavior.ESCAPE, trigger="CRITICAL_THREAT", policy="threat_escape_policy")
            return self.current_behavior

        # 4. Cấp ưu tiên 4: Safety Recovery với Hysteresis
        # Kích hoạt khi vật cản < 0.22m, duy trì RECOVERY dứt khoát cho tới khi khoảng cách mở rộng >= 0.35m
        is_recovering = (self.current_behavior == Behavior.RECOVERY)
        recovery_thresh = 0.35 if is_recovering else 0.22
        if obstacle_dist_m > 0 and obstacle_dist_m < recovery_thresh:
            self._transition_to(Behavior.RECOVERY, trigger="OBSTACLE_PROXIMITY", policy="obstacle_recovery_policy")
            return self.current_behavior

        # 5. Kiểm tra Hysteresis & Minimum Dwell Time (2.5 - 3.0s):
        # Nếu chưa đủ thời gian chạy tối thiểu và không bị ngắt khẩn cấp -> giữ nguyên hành vi
        if self.active_time_sec < self.min_active_time_sec:
            return self.current_behavior

        # 6. Đánh giá chuyển trạng thái tương tác & động lực nội tại với Dual-Threshold Hysteresis
        candidate = Behavior.IDLE
        cand_trigger = "IDLE_QUIET"
        cand_policy = "autonomous_idle_policy"
        is_exploring = (self.current_behavior == Behavior.EXPLORE)

        # Dynamic Target Prioritization (Ưu tiên mục tiêu có trọng số):
        # Khi cả người và bóng cùng xuất hiện trong phòng:
        # - Nếu social_need < 0.20 (đã thỏa mãn tương tác xã hội) VÀ bóng hiện diện (ball_confidence >= 0.70):
        #   Robot ưu tiên bóng để giải tỏa buồn chán (boredom relief), tránh hiện tượng phân vân giật đầu
        # - Nếu social_need >= 0.20:
        #   Robot tiếp tục duy trì ưu tiên tương tác xã hội (SOCIAL)
        has_both_targets = (human_engagement >= 0.60 and ball_confidence >= 0.70)
        prefer_ball_over_human = has_both_targets and (social_need < 0.20 or boredom > 0.45)

        if ball_confidence >= 0.75 and prefer_ball_over_human:
            if target_dist > 0.45:
                candidate = Behavior.APPROACH
                cand_trigger = "BOREDOM_BALL_PRIORITY"
                cand_policy = "ball_approach_policy"
            else:
                candidate = Behavior.BALL
                cand_trigger = "BALL_INTERACT"
                cand_policy = "ball_interaction_policy"
        elif human_engagement >= 0.70:
            candidate = Behavior.SOCIAL
            cand_trigger = "HUMAN_ENGAGEMENT"
            cand_policy = "social_attention_policy"
        elif ball_confidence >= 0.75:
            if target_dist > 0.45:
                candidate = Behavior.APPROACH
                cand_trigger = "BALL_APPROACH"
                cand_policy = "ball_approach_policy"
            else:
                candidate = Behavior.BALL
                cand_trigger = "BALL_INTERACT"
                cand_policy = "ball_interaction_policy"
        elif target_visible:
            if target_dist > 0.85:
                candidate = Behavior.APPROACH
                cand_trigger = "TARGET_APPROACH"
                cand_policy = "target_approach_policy"
            else:
                candidate = Behavior.TRACK
                cand_trigger = "TARGET_TRACKING"
                cand_policy = "target_tracking_policy"
        elif is_exploring:
            # Hysteresis Exit: Duy trì EXPLORE cho đến khi tò mò giảm dưới 0.35 VÀ buồn chán dưới 0.25
            if curiosity > self.explore_exit_threshold or boredom > 0.25:
                candidate = Behavior.EXPLORE
                cand_trigger = "EXPLORE_MAINTAIN"
                cand_policy = "curiosity_explore_policy"
            elif boredom > 0.15:
                candidate = Behavior.LOOK_AROUND
                cand_trigger = "BOREDOM_SCAN"
                cand_policy = "look_around_policy"
            else:
                candidate = Behavior.IDLE
                cand_trigger = "IDLE_QUIET"
                cand_policy = "autonomous_idle_policy"
        else:
            # Hysteresis Enter: Chỉ chuyển sang EXPLORE khi tò mò >= 0.65 HOẶC buồn chán >= 0.50
            if curiosity >= self.explore_enter_threshold or boredom >= 0.50:
                candidate = Behavior.EXPLORE
                cand_trigger = "CURIOSITY_EXPLORE" if curiosity >= self.explore_enter_threshold else "BOREDOM_EXPLORE"
                cand_policy = "curiosity_explore_policy"
            elif boredom > 0.20:
                candidate = Behavior.LOOK_AROUND
                cand_trigger = "BOREDOM_SCAN"
                cand_policy = "look_around_policy"
            else:
                candidate = Behavior.IDLE
                cand_trigger = "IDLE_QUIET"
                cand_policy = "autonomous_idle_policy"

        # Áp dụng Cooldown trước khi đổi hành vi
        if candidate != self.current_behavior:
            if self.cooldown_timer_sec <= 0:
                self._transition_to(candidate, trigger=cand_trigger, policy=cand_policy)
        else:
            self.behavior_trigger = cand_trigger
            self.active_policy = cand_policy

        return self.current_behavior

    def _transition_to(self, new_behavior: Behavior, trigger: str = "TRANSITION", policy: str = "default_policy"):
        if self.current_behavior != new_behavior:
            if self.active_time_sec > 0:
                self.behavior_durations.append(float(self.active_time_sec))
            self.last_behavior = self.current_behavior
            self.current_behavior = new_behavior
            self.behavior_trigger = trigger
            self.active_policy = policy
            self.active_time_sec = 0.0
            self.cooldown_timer_sec = self.cooldown_sec
            self.switch_counter += 1
            self.last_transition_switched = True
        else:
            self.last_transition_switched = False
            self.cooldown_timer_sec = self.cooldown_sec

    def sanitize_action(self,
                        linear_vel: float,
                        angular_vel: float,
                        pan_vel: float,
                        tilt_vel: float,
                        attention_intensity: float,
                        current_pan_rad: float,
                        current_tilt_rad: float,
                        front_obstacle_m: float,
                        front_obstacle_x: float = 0.0,
                        target_dist: float = 2.0,
                        gaze_duration: float = 0.0,
                        acoustic_active: bool = False,
                        acoustic_direction: float = 0.0,
                        dt: float = 0.1,
                        boredom: float = 0.0,
                        target_is_human: bool = True,
                        enable_behavior_reflexes: bool = True) -> Tuple[np.ndarray, bool, str]:
        """
        Deterministic Safety Layer (Doc 07 & Upgrade Plan Section 13).
        Bộ lọc an toàn bắt buộc đứng sau Policy:
        - Giới hạn vận tốc bánh xích & gia tốc (Slew Rate Limiter)
        - Giới hạn hành trình & vận tốc servo Pan/Tilt
        - Khẩn cấp dừng tiến nếu phía trước có vật cản gần (Obstacle Proximity Clamp)
        - Ngăn chặn lùi mù không kiểm soát (Blind Reverse Prevention)
        - Phản xạ Anti-Stare Look-Away Reflex giải phóng điểm nhìn
        - Phản xạ định hướng âm thanh thoáng qua (Acoustic Orienting Reflex)
        - Phản xạ quét góc nhìn khám phá khi môi trường tĩnh (Exploratory Scan Reflex)
        - Ghi nhận telemetry so sánh Raw Action vs Safe Action
        """
        self.total_commands_evaluated += 1
        overridden = False
        reason = "NONE"

        track_half_width = 0.07  # Khoảng cách nửa bề rộng vệt xích T-RAX
        self.last_raw_action = np.array([linear_vel, angular_vel, pan_vel, tilt_vel, attention_intensity], dtype=np.float32)
        self.last_raw_left = float(linear_vel - angular_vel * track_half_width)
        self.last_raw_right = float(linear_vel + angular_vel * track_half_width)

        # Decrement persistent avoidance steering timer
        if self.avoid_steer_timer > 0.0:
            self.avoid_steer_timer = max(0.0, self.avoid_steer_timer - dt)
        if self.avoid_steer_timer == 0.0:
            self.avoid_steer_dir = 0.0

        # Safe mechanical bounds (rad)
        PAN_MIN, PAN_MAX = -1.047, 1.047   # [-60 deg .. +60 deg]
        TILT_MIN, TILT_MAX = -0.698, 0.698 # [-40 deg .. +40 deg]

        # Max velocities
        MAX_LIN = 0.50   # m/s
        MAX_ANG = 1.50   # rad/s
        MAX_SERVO_VEL = 2.0  # rad/s

        safe_lin = float(np.clip(linear_vel, -MAX_LIN, MAX_LIN))
        safe_ang = float(np.clip(angular_vel, -MAX_ANG, MAX_ANG))
        safe_pan_v = float(np.clip(pan_vel, -MAX_SERVO_VEL, MAX_SERVO_VEL))
        safe_tilt_v = float(np.clip(tilt_vel, -MAX_SERVO_VEL, MAX_SERVO_VEL))
        safe_attn = float(np.clip(attention_intensity, 0.0, 1.0))
        corrections = []
        previous_stage = self.last_raw_action.copy()
        def record(stage):
            nonlocal previous_stage
            current = np.array([safe_lin, safe_ang, safe_pan_v, safe_tilt_v, safe_attn])
            if not np.allclose(current, previous_stage, atol=1e-6, rtol=0):
                corrections.append(stage)
            previous_stage = current
        record("SPEED_CLIP")

        # 0. State Machine Arbitration (Doc 07 Section 3 & Doc 13 Section 5)
        # Trong trạng thái IDLE (không có mục tiêu, không có người, không có âm thanh),
        # đế bánh xích phải đứng yên tuyệt đối (v = 0, omega = 0) để duy trì tĩnh lặng cơ học và tiết kiệm pin.
        if self.current_behavior == Behavior.IDLE:
            safe_lin = 0.0
            safe_ang = 0.0
        elif self.current_behavior == Behavior.RECOVERY:
            if safe_lin > 0:
                safe_lin = 0.0
        record("BEHAVIOR_HOLD")

        # Motor Deadband Filter (Hardware friction model)
        if abs(safe_lin) < 0.03:
            safe_lin = 0.0
        if abs(safe_ang) < 0.03:
            safe_ang = 0.0
        record("MOTOR_DEADBAND")

        # 1. Acoustic Orienting Reflex (Doc 13 Section 9)
        if acoustic_active:
            target_acoustic_pan = -float(acoustic_direction)
            err = target_acoustic_pan - current_pan_rad
            safe_pan_v = float(np.clip(err * 2.0, -MAX_SERVO_VEL, MAX_SERVO_VEL))
        record("ACOUSTIC_ORIENT")

        # 2. Anti-stare Disengagement Reflex (Doc 05, Doc 07, Doc 13 Section 7 & 10)
        if enable_behavior_reflexes and self.disengage_timer > 0:
            self.disengage_timer = max(0.0, self.disengage_timer - dt)
            safe_pan_v = self.disengage_direction
        elif enable_behavior_reflexes and gaze_duration > 2.6:
            if abs(safe_pan_v) < 0.25:
                # Kích hoạt phản xạ quay mắt đi kéo dài 0.8s
                self.disengage_timer = 0.8
                self.disengage_direction = 0.85 if current_pan_rad <= 0 else -0.85
                safe_pan_v = self.disengage_direction

        # 3. Autonomous Exploratory Scan Reflex (chỉ kích hoạt khi đang ở hành vi LOOK_AROUND)
        elif enable_behavior_reflexes and self.current_behavior == Behavior.LOOK_AROUND and not acoustic_active and gaze_duration == 0.0 and boredom > 0.20:
            if current_pan_rad >= 0.70:
                self.scan_direction = -0.75
            elif current_pan_rad <= -0.70:
                self.scan_direction = 0.75
            safe_pan_v = self.scan_direction
        record("GAZE_OR_SCAN_REFLEX")

        # 4. Obstacle Stop Override & Latched Avoidance Steering (Doc 07 Section 4, Doc 13 Section 5)
        # Khắc phục hiện tượng chattering / quay left-right tại chỗ:
        # Hướng né tránh được chốt (latched) kiên định tối thiểu 1.0s để robot quay dứt khoát ra khỏi vật cản/tường.
        if 0.0 < front_obstacle_m < 0.25:
            # Nếu chưa chốt hướng rẽ hoặc đã hết timer, chọn hướng thoát theo vị trí vật cản
            if self.avoid_steer_dir == 0.0:
                self.avoid_steer_dir = -1.0 if front_obstacle_x <= 0.0 else 1.0
                self.avoid_steer_timer = 1.0  # Chốt hướng rẽ trong 1.0s

            # Phát hiện kẹt (Stuck Detection): Nếu ở quá sát vật cản (< 0.20m) nhiều chu kỳ, kích hoạt lùi cứu nguy
            if front_obstacle_m < 0.20:
                self.stuck_counter += 1
                if self.stuck_counter >= 4:
                    safe_lin = -0.12  # Lùi nhẹ để tạo khoảng trống thoát kẹt
                    overridden = True
                    reason = "OBSTACLE_BACKOFF_ESCAPE"
                elif safe_lin > 0.0:
                    safe_lin = 0.0
                    overridden = True
                    reason = "OBSTACLE_PROXIMITY_CLAMP"
            else:
                self.stuck_counter = 0
                if safe_lin > 0.0:
                    safe_lin = 0.0
                    overridden = True
                    reason = "OBSTACLE_PROXIMITY_CLAMP"

            # Hướng dẫn lái né tránh mượt mà (smooth avoidance steering)
            target_steer = self.avoid_steer_dir * 0.65
            if (self.avoid_steer_dir > 0 and safe_ang < 0.3) or (self.avoid_steer_dir < 0 and safe_ang > -0.3):
                safe_ang = target_steer

        # b) Giảm tốc mượt mà & rẽ né tránh từ xa khi cự ly trong khoảng [0.25m .. 0.45m] (Avoidance Reflex)
        elif 0.25 <= front_obstacle_m < 0.45:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            max_lin_allow = MAX_LIN * float(np.clip((front_obstacle_m - 0.22) / 0.23, 0.0, 1.0))
            if safe_lin > max_lin_allow:
                safe_lin = max_lin_allow

            # Né tránh từ xa: duy trì hướng rẽ kiên định nếu đang trong timer
            if self.avoid_steer_dir != 0.0:
                target_steer = self.avoid_steer_dir * 0.45
                if (self.avoid_steer_dir > 0 and safe_ang < 0.2) or (self.avoid_steer_dir < 0 and safe_ang > -0.2):
                    safe_ang = target_steer
            elif abs(front_obstacle_x) < 0.30:
                self.avoid_steer_dir = -1.0 if front_obstacle_x <= 0.0 else 1.0
                self.avoid_steer_timer = 0.8
                safe_ang = self.avoid_steer_dir * 0.45

        else:
            if self.avoid_steer_timer <= 0:
                self.stuck_counter = 0
        record("OBSTACLE_AVOIDANCE")

        # 5. Blind Reverse Prevention (Doc 07, Doc 13 & Embedded Safety)
        # Robot chỉ có ToF và camera phía trước, không có cảm biến phía sau.
        # Cho phép lùi khi:
        #   a) Phía trước có vật cản (< 0.35m) cần lùi cứu nguy
        #   b) CON NGƯỜI áp sát trong vùng không gian cá nhân (< 1.20m - Doc 10) cần lùi giữ khoảng cách.
        #      Không áp dụng cho bóng/đồ vật: lùi tránh bóng là lùi mù vô cớ (RC1 trong phân tích log S7).
        #   c) Kích hoạt phản xạ lùi thoát kẹt (OBSTACLE_BACKOFF_ESCAPE)
        # Âm thanh thoáng qua KHÔNG còn cho phép lùi mù: phản xạ âm thanh chỉ là quay đầu (pan) định hướng.
        # Ngưỡng -0.03 trùng với motor deadband để không còn khe hở [-0.06, -0.03) lọt lệnh lùi chậm (RC1b).
        if safe_lin < -0.03:
            allow_reverse = ((0.0 < front_obstacle_m < 0.35)
                             or (target_is_human and 0.0 < target_dist < 1.20)
                             or (reason == "OBSTACLE_BACKOFF_ESCAPE"))
            if not allow_reverse:
                safe_lin = 0.0
                reason = "BLIND_REVERSE_CLAMP"
                if linear_vel < -0.30:
                    overridden = True
        record("BLIND_REVERSE_CLAMP")
        reverse_blocked = reason == "BLIND_REVERSE_CLAMP"

        # 6. Pan Servo Range Protection
        if current_pan_rad <= PAN_MIN and safe_pan_v < 0:
            safe_pan_v = 0.0
            overridden = True
            reason = "SERVO_PAN_MIN_LIMIT"
        elif current_pan_rad >= PAN_MAX and safe_pan_v > 0:
            safe_pan_v = 0.0
            overridden = True
            reason = "SERVO_PAN_MAX_LIMIT"

        # 7. Tilt Servo Range Protection
        if current_tilt_rad <= TILT_MIN and safe_tilt_v < 0:
            safe_tilt_v = 0.0
            overridden = True
            reason = "SERVO_TILT_MIN_LIMIT"
        elif current_tilt_rad >= TILT_MAX and safe_tilt_v > 0:
            safe_tilt_v = 0.0
            overridden = True
            reason = "SERVO_TILT_MAX_LIMIT"
        record("SERVO_RANGE")

        # 8. Slew Rate Limiter (Chống rung giật cơ cấu truyền động & triệt tiêu Jerk)
        MAX_LIN_RAMP = 1.5 * dt    # 0.15 m/s ramp per step (max accel 1.5 m/s^2)
        MAX_ANG_RAMP = 2.0 * dt    # 0.20 rad/s ramp per step
        MAX_SERVO_RAMP = 1.6 * dt  # 0.16 rad/s ramp per step for pan/tilt servos

        # Giảm tốc an toàn khi có cản mà vẫn khống chế biến thiên đột ngột (Jerk <= 0.05)
        if reason in ("OBSTACLE_PROXIMITY_CLAMP", "OBSTACLE_BACKOFF_ESCAPE") and safe_lin <= 0.0:
            safe_lin = max(safe_lin, self.last_safe_lin - 3.0 * dt)
        else:
            safe_lin = float(np.clip(safe_lin, self.last_safe_lin - MAX_LIN_RAMP, self.last_safe_lin + MAX_LIN_RAMP))
        self.last_safe_lin = safe_lin

        safe_ang = float(np.clip(safe_ang, self.last_safe_ang - MAX_ANG_RAMP, self.last_safe_ang + MAX_ANG_RAMP))
        safe_pan_v = float(np.clip(safe_pan_v, self.last_safe_pan_v - MAX_SERVO_RAMP, self.last_safe_pan_v + MAX_SERVO_RAMP))
        safe_tilt_v = float(np.clip(safe_tilt_v, self.last_safe_tilt_v - MAX_SERVO_RAMP, self.last_safe_tilt_v + MAX_SERVO_RAMP))
        record("SLEW_LIMIT")
        # Hard constraints must survive the smoothing stage. In particular a
        # previous reverse command must never bleed into a newly blocked reverse.
        if reverse_blocked or self.current_behavior in (Behavior.IDLE, Behavior.SAFE_STOP):
            safe_lin = 0.0
        if self.current_behavior == Behavior.SAFE_STOP:
            safe_ang = safe_pan_v = safe_tilt_v = 0.0
        if 0 < front_obstacle_m < .25 and safe_lin > 0:
            safe_lin = 0.0
        if current_pan_rad <= PAN_MIN and safe_pan_v < 0 or current_pan_rad >= PAN_MAX and safe_pan_v > 0:
            safe_pan_v = 0.0
        if current_tilt_rad <= TILT_MIN and safe_tilt_v < 0 or current_tilt_rad >= TILT_MAX and safe_tilt_v > 0:
            safe_tilt_v = 0.0
        record("HARD_CONSTRAINT")
        self.last_safe_lin = safe_lin

        self.last_safe_ang = safe_ang
        self.last_safe_pan_v = safe_pan_v
        self.last_safe_tilt_v = safe_tilt_v

        # Speed clamp check (Upgrade Plan Section 3.4 & 13)
        if not overridden and (abs(linear_vel) > MAX_LIN or abs(angular_vel) > MAX_ANG):
            overridden = True
            reason = "SPEED_CLAMP"

        if overridden:
            self.total_overrides_triggered += 1
            self.last_override_reason = reason

        self.last_intervention = reason if overridden else "NONE"
        track_half_width = 0.07
        self.last_safe_action = np.array([safe_lin, safe_ang, safe_pan_v, safe_tilt_v, safe_attn], dtype=np.float32)
        self.last_corrections = corrections
        for correction in corrections:
            self.correction_counts[correction] = self.correction_counts.get(correction, 0) + 1
        if not np.allclose(self.last_raw_action, self.last_safe_action, atol=1e-6, rtol=0):
            self.action_changed_count += 1
        self.last_safe_left = float(safe_lin - safe_ang * track_half_width)
        self.last_safe_right = float(safe_lin + safe_ang * track_half_width)

        safe_action = self.last_safe_action
        return safe_action, overridden, reason

    def get_safety_metrics(self) -> dict:
        override_rate = (self.total_overrides_triggered / max(1, self.total_commands_evaluated))
        durations = self.behavior_durations + ([float(self.active_time_sec)] if self.active_time_sec > 0 else [])
        mean_dur = float(np.mean(durations)) if len(durations) > 0 else float(self.active_time_sec)
        return {
            "current_behavior": self.current_behavior.name,
            "trigger": self.behavior_trigger,
            "active_policy": self.active_policy,
            "active_time_sec": float(self.active_time_sec),
            "safety_override_rate": float(override_rate),
            "total_overrides": int(self.total_overrides_triggered),
            "action_changed_rate": self.action_changed_count / max(1, self.total_commands_evaluated),
            "correction_reasons": self.last_corrections.copy(),
            "correction_counts": self.correction_counts.copy(),
            "last_override_reason": self.last_override_reason,
            "intervention": self.last_intervention,
            "intervention_type": self.last_intervention,
            "switch_count": int(self.switch_counter),
            "mean_behavior_duration": mean_dur,
            "raw_left": float(self.last_raw_left),
            "raw_right": float(self.last_raw_right),
            "safe_left": float(self.last_safe_left),
            "safe_right": float(self.last_safe_right),
            "raw_action": self.last_raw_action.tolist(),
            "safe_action": self.last_safe_action.tolist()
        }
