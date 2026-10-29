import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

class BehavioralScenario(str, Enum):
    S0_MOTOR_SANITY = "S0"
    S1_VISUAL_EXPLORATION = "S1"
    S2_OBJECT_CURIOSITY = "S2"
    S3_DYNAMIC_OBJECT = "S3"
    S4_ACOUSTIC_EVENT = "S4"
    S5_HUMAN_PRESENCE = "S5"
    S6_ATTENTION_SEEKING = "S6"
    S7_MIXED = "S7"

@dataclass
class SafeCommand:
    linear_velocity: float
    angular_velocity: float
    pan_velocity: float
    tilt_velocity: float
    attention_intensity: float
    overridden: bool = False
    override_reason: Optional[str] = None
    safety_level: str = "normal"  # "normal", "limited", "stop"

@dataclass
class TrackingRequest:
    target_id: str
    target_type: str  # "dynamic_object", "human", "object"
    initial_bearing: float
    confidence: float
    max_duration: float = 5.0
    handoff_reason: str = "interest_threshold_exceeded"

@dataclass
class TrackingStatus:
    active: bool = False
    target_visible: bool = False
    target_confidence: float = 0.0
    elapsed_time: float = 0.0
    lost_target_time: float = 0.0
    should_release: bool = False

@dataclass
class InterestSignal:
    active: bool = False
    type: str = "none"  # "none", "dynamic_object", "human", "acoustic_event", "novel_object", "attention_needed"
    score: float = 0.0
    target_bearing: Optional[float] = None
    target_id: Optional[str] = None
    ttl: float = 2.0

class ScenarioManager:
    """
    Scenario Manager & Lifecycle Arbiter theo tài liệu 13_behavioral_scenarios.md.
    Quản lý cấu hình, phát sinh sự kiện, điều phối handoff và đo lường metrics
    cho toàn bộ chuỗi kịch bản S0 đến S7:
      S0: Motor Sanity (Empty room, low jerk, stability, stillness)
      S1: Visual Exploration (View novelty, scan, spatial coverage)
      S2: Object Curiosity (Static objects, k-NN novelty, anti-stare)
      S3: Dynamic Object (Ball rolling, motion detection, TrackingRequest handoff)
      S4: Acoustic Event (Sound impulses, refractory cooldown, orientation)
      S5: Human Presence (Human gaze, mutual attention, look-away)
      S6: Attention Seeking (5-tier escalation, signal cost, anti-spam)
      S7: Mixed Environment (Multi-event integration & runtime arbitration)
    """
    def __init__(self, scenario: str = "S7"):
        self.scenario = self._parse_scenario(scenario)
        self.reset()

    def _parse_scenario(self, scenario_str: str) -> BehavioralScenario:
        clean_str = scenario_str.upper().strip()
        for sc in BehavioralScenario:
            if clean_str in (sc.value, sc.name, f"SCENARIO_{sc.value}"):
                return sc
        if "MIXED" in clean_str:
            return BehavioralScenario.S7_MIXED
        return BehavioralScenario.S7_MIXED

    def reset(self):
        self.step_idx = 0
        self.sim_time = 0.0

        # TrackingPolicy handoff state (Doc 13 Section 2.4)
        self.active_tracking_request: Optional[TrackingRequest] = None
        self.tracking_status = TrackingStatus()
        self.tracking_handoff_count = 0
        self.tracking_timeouts = 0
        self.tracking_cooldown = 0.0

        # Acoustic event state (Doc 13 Section 9)
        self.acoustic_active = False
        self.acoustic_level = 0.0
        self.acoustic_direction = 0.0
        self.acoustic_cooldown = 0.0
        self.acoustic_timer = 0.0
        self.acoustic_event_count = 0
        self.acoustic_responses = 0
        self.acoustic_position = None

        # Inspection & Gaze state (Doc 13 Section 7 & 10)
        self.current_inspected_object_id: Optional[str] = None
        self.object_inspection_duration = 0.0
        self.object_inspections = 0
        self.human_gaze_duration = 0.0
        self.human_orientations = 0
        self.stare_violations = 0

        # Motion & Stability tracking (Doc 13 Section 5 & 6)
        self.jerk_accum = 0.0
        self.stillness_steps = 0
        self.saturation_steps = 0
        self.total_overrides = 0
        self.recent_actions: List[np.ndarray] = []
        self.view_angles: List[Tuple[float, float]] = []

        # Attention Seeking tracking (Doc 13 Section 11)
        self.attention_attempts = 0
        self.attention_successes = 0
        self.signal_costs_accum = 0.0
        self.signal_spam_steps = 0
        self.escalation_levels: List[float] = []
        self.previous_tier = 0
        self.attention_response_window = 0.0
        self.last_attention_time = -10.0
        self.inspected_ids = set()
        self.last_target_id = None
        self.tracking_motion_steps = self.tracking_aligned_steps = 0
        self.tracking_error_sum = 0.0
        self.collision_steps = self.unstable_steps = 0

    def trigger_acoustic_event(self, sound_pos: np.ndarray, duration: float = 1.5):
        """
        Kích hoạt sự kiện âm thanh thoáng qua (sound transient) theo Doc 13 Section 9.
        """
        if self.acoustic_cooldown <= 0:
            self.acoustic_position = np.asarray(sound_pos).copy()
            self.acoustic_active = True
            self.acoustic_level = 0.85
            # sound azimuth relative to robot body
            self.acoustic_direction = float(np.arctan2(sound_pos[0], sound_pos[1]))
            self.acoustic_timer = duration
            self.acoustic_event_count += 1

    def configure_environment(self, env) -> Dict[str, Any]:
        """
        Cấu hình môi trường vật lý & sub-engines khớp chuẩn với từng Scenario (Doc 13 Section 5-12).
        """
        config = {
            "has_human": False,
            "has_static_objects": False,
            "has_dynamic_motion": False,
            "has_acoustic_events": False,
            "target_pos": np.array([5.0, 5.0, 0.08]),
            "description": ""
        }

        sc = self.scenario

        if sc == BehavioralScenario.S0_MOTOR_SANITY:
            config["description"] = "S0: Motor Sanity (Empty room, stability, stillness, zero jerk)"
            env.human_agent.is_present = False
            env.data.mocap_pos[env.target_mocap_id] = np.array([5.0, 5.0, 0.08])
            env.event_scheduler.scheduled_events.clear()

        elif sc == BehavioralScenario.S1_VISUAL_EXPLORATION:
            config["description"] = "S1: Visual Exploration (Scan, view diversity, no static/dynamic targets)"
            env.human_agent.is_present = False
            env.data.mocap_pos[env.target_mocap_id] = np.array([5.0, 5.0, 0.08])
            env.event_scheduler.scheduled_events.clear()

        elif sc == BehavioralScenario.S2_OBJECT_CURIOSITY:
            config["description"] = "S2: Object Curiosity (Static object inspection, novelty decay, anti-stare)"
            config["has_static_objects"] = True
            env.human_agent.is_present = False
            # Đặt mục tiêu tĩnh trên mặt sàn ở cự ly quan sát tối ưu (0.0, 1.2m, 0.08m)
            config["target_pos"] = np.array([0.0, 1.2, 0.08])
            env.data.mocap_pos[env.target_mocap_id] = config["target_pos"]
            env.event_scheduler.scheduled_events.clear()

        elif sc == BehavioralScenario.S3_DYNAMIC_OBJECT:
            config["description"] = "S3: Dynamic Object (Ball rolling, motion detection, TrackingPolicy handoff)"
            config["has_dynamic_motion"] = True
            env.human_agent.is_present = False
            config["target_pos"] = np.array([-1.0, 1.2, 0.08], dtype=np.float32)
            env.data.mocap_pos[env.target_mocap_id] = config["target_pos"]
            # Tạo chuỗi chuyển động bóng lăn liên tục qua lại trên mặt sàn (Z = 0.08m)
            env.event_scheduler.scheduled_events = [
                {"time": 1.0, "type": "ball_roll", "start_pos": np.array([-1.1, 1.2, 0.08], dtype=np.float32), "end_pos": np.array([1.1, 1.2, 0.08], dtype=np.float32), "pos": np.array([-1.1, 1.2, 0.08], dtype=np.float32), "duration": 3.0, "triggered": False},
                {"time": 5.0, "type": "ball_roll", "start_pos": np.array([1.1, 1.3, 0.08], dtype=np.float32), "end_pos": np.array([-1.1, 1.3, 0.08], dtype=np.float32), "pos": np.array([1.1, 1.3, 0.08], dtype=np.float32), "duration": 3.0, "triggered": False},
                {"time": 9.0, "type": "ball_roll", "start_pos": np.array([-0.9, 1.5, 0.08], dtype=np.float32), "end_pos": np.array([0.9, 1.1, 0.08], dtype=np.float32), "pos": np.array([-0.9, 1.5, 0.08], dtype=np.float32), "duration": 3.0, "triggered": False},
                {"time": 13.0, "type": "ball_roll", "start_pos": np.array([0.9, 1.1, 0.08], dtype=np.float32), "end_pos": np.array([-0.9, 1.4, 0.08], dtype=np.float32), "pos": np.array([0.9, 1.1, 0.08], dtype=np.float32), "duration": 3.0, "triggered": False},
            ]

        elif sc == BehavioralScenario.S4_ACOUSTIC_EVENT:
            config["description"] = "S4: Acoustic Event (Sound transients, orientation, refractory cooldown)"
            config["has_acoustic_events"] = True
            env.human_agent.is_present = False
            env.data.mocap_pos[env.target_mocap_id] = np.array([5.0, 5.0, 0.08])
            env.event_scheduler.scheduled_events = [
                {"time": 2.0, "type": "sound_cue", "pos": np.array([1.0, 1.0, 0.5]), "duration": 1.5, "triggered": False},
                {"time": 5.0, "type": "sound_cue", "pos": np.array([-1.0, 1.0, 0.5]), "duration": 1.5, "triggered": False},
                {"time": 8.0, "type": "sound_cue", "pos": np.array([0.5, 1.5, 0.5]), "duration": 1.5, "triggered": False},
            ]

        elif sc == BehavioralScenario.S5_HUMAN_PRESENCE:
            config["description"] = "S5: Human Presence (Human gaze, mutual attention, look-away, anti-stare)"
            config["has_human"] = True
            config["target_pos"] = np.array([0.0, 1.2, 0.0])
            env.human_agent.is_present = True
            env.human_agent.position = config["target_pos"].copy()
            env.data.mocap_pos[env.target_mocap_id] = np.array([0.0, 1.2, 0.375], dtype=np.float32)
            env.event_scheduler.scheduled_events.clear()

        elif sc == BehavioralScenario.S6_ATTENTION_SEEKING:
            config["description"] = "S6: Attention Seeking (5-tier intensity escalation, anti-spam, cost balance)"
            config["has_human"] = True
            config["target_pos"] = np.array([0.2, 1.1, 0.0])
            env.human_agent.is_present = True
            env.human_agent.position = config["target_pos"].copy()
            env.data.mocap_pos[env.target_mocap_id] = np.array([0.2, 1.1, 0.375], dtype=np.float32)
            env.event_scheduler.scheduled_events.clear()

        else: # S7_MIXED
            config["description"] = "S7: Mixed Environment (Full multi-event integration & runtime arbitration)"
            config["has_human"] = True
            config["target_pos"] = np.array([0.0, 1.2, 0.0])
            config["has_static_objects"] = True
            config["has_dynamic_motion"] = True
            config["has_acoustic_events"] = True
            env.human_agent.is_present = True
            env.event_scheduler.generate_episode_events(episode_length_sec=env.max_episode_steps * 0.1)

        return config

    def update_step(self,
                    dt: float,
                    action: np.ndarray,
                    safe_action: np.ndarray,
                    overridden: bool,
                    override_reason: str,
                    pan_deg: float,
                    tilt_deg: float,
                    target_visible: bool,
                    target_bearing: float,
                    target_distance: float,
                    human_aligned: bool,
                    human_responded: bool,
                    tier: int,
                    is_spam: bool,
                    novelty_score: float,
                    target_id: str = "target",
                    target_is_human: bool = False,
                    target_motion: bool = False,
                    gaze_error: float = 0.0,
                    collision: bool = False,
                    simulation_unstable: bool = False) -> Dict[str, Any]:
        """
        Cập nhật metrics thời gian thực theo chuỗi quy tắc của tài liệu 13.
        """
        self.step_idx += 1
        self.sim_time += dt
        self.view_angles.append((pan_deg, tilt_deg))
        self.collision_steps += int(collision)
        self.unstable_steps += int(simulation_unstable)
        if target_id != self.last_target_id:
            self.object_inspection_duration = self.human_gaze_duration = 0.0
            self.last_target_id = target_id

        # 1. Đo lường Jerk & Ổn định cơ học (S0)
        action_energy = float(np.sum(np.abs(action[:4])))
        if len(self.recent_actions) > 0:
            diff = action[:4] - self.recent_actions[-1][:4]
            self.jerk_accum += float(np.sum(np.square(diff)))

        self.recent_actions.append(action.copy())
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)

        if action_energy < 0.08 and action[4] < .2:
            self.stillness_steps += 1

        if abs(pan_deg - 90.0) >= 59.0 or abs(tilt_deg - 90.0) >= 39.0:
            self.saturation_steps += 1

        if overridden:
            self.total_overrides += 1

        # 2. Handoff TrackingPolicy cho Dynamic Object (S3)
        if self.tracking_cooldown > 0:
            self.tracking_cooldown = max(0.0, self.tracking_cooldown - dt)

        if self.scenario in (BehavioralScenario.S3_DYNAMIC_OBJECT, BehavioralScenario.S7_MIXED):
            if target_motion and not target_is_human:
                self.tracking_motion_steps += 1
                self.tracking_aligned_steps += int(target_visible and abs(gaze_error) < .25)
                self.tracking_error_sum += abs(gaze_error)
            if target_motion and not target_is_human and target_visible and not self.tracking_status.active and self.tracking_cooldown <= 0:
                # Kích hoạt TrackingRequest handoff (Doc 13 Section 2.4)
                self.active_tracking_request = TrackingRequest(
                    target_id=target_id,
                    target_type="dynamic_object",
                    initial_bearing=target_bearing,
                    confidence=0.88,
                    max_duration=3.5,
                    handoff_reason="dynamic_motion_detected"
                )
                self.tracking_status = TrackingStatus(
                    active=True,
                    target_visible=True,
                    target_confidence=0.88,
                    elapsed_time=0.0
                )
                self.tracking_handoff_count += 1
            elif self.tracking_status.active:
                self.tracking_status.elapsed_time += dt
                self.tracking_status.target_visible = target_visible
                same_target = self.active_tracking_request.target_id == target_id and not target_is_human
                if not target_visible or not same_target:
                    self.tracking_status.lost_target_time += dt
                else:
                    self.tracking_status.lost_target_time = 0.0

                # Kiểm tra Timeout & Release condition (Doc 13 Section 2.4)
                if self.tracking_status.lost_target_time > 1.2:
                    # Mất dấu mục tiêu quá lâu -> Timeout failure
                    self.tracking_status.active = False
                    self.tracking_status.should_release = True
                    self.tracking_timeouts += 1
                    self.tracking_cooldown = 1.5
                elif (same_target and not target_motion) or self.tracking_status.elapsed_time >= self.active_tracking_request.max_duration:
                    # Hoàn thành phiên theo dõi -> Graceful release
                    self.tracking_status.active = False
                    self.tracking_status.should_release = True
                    self.tracking_cooldown = 1.5

        # 3. Phản ứng Acoustic Event (S4)
        if self.acoustic_active:
            self.acoustic_timer -= dt
            # Kiểm tra xem robot có quay đầu về hướng âm thanh không
            pan_rad = np.deg2rad(pan_deg - 90.0)
            if abs(-self.acoustic_direction - pan_rad) < 0.35:
                self.acoustic_responses += 1
                self.acoustic_active = False
                self.acoustic_cooldown = 2.0  # Refractory cooldown (Doc 13 Section 9)
            elif self.acoustic_timer <= 0:
                self.acoustic_active = False
                self.acoustic_cooldown = 1.5
        elif self.acoustic_cooldown > 0:
            self.acoustic_cooldown = max(0.0, self.acoustic_cooldown - dt)

        # 4. Gaze & Anti-stare evaluation (S2 & S5)
        if target_visible:
            # Sustained fixation during approach or moving-object tracking is
            # purposeful; stare measures stationary, unbroken attention.
            if human_aligned and np.sum(np.abs(action[:2])) < .08 and not target_motion:
                self.human_gaze_duration += dt
                if target_is_human and self.human_gaze_duration > 0.3 and self.human_gaze_duration - dt <= 0.3:
                    self.human_orientations += 1
                if self.human_gaze_duration > 3.5:
                    self.stare_violations += 1
            else:
                self.human_gaze_duration = 0.0

            if (not target_is_human and abs(gaze_error) < .2 and .35 <= target_distance <= .8
                    and np.sum(np.abs(action[:2])) < .08 and not target_motion):
                self.object_inspection_duration += dt
                if self.object_inspection_duration >= 1.0 and target_id not in self.inspected_ids:
                    self.inspected_ids.add(target_id)
                    self.object_inspections += 1
            else:
                self.object_inspection_duration = 0.0
        else:
            self.human_gaze_duration = 0.0
            self.object_inspection_duration = 0.0

        # 5. Attention Escalation & Spam tracking (S6)
        attn_val = float(action[4])
        self.signal_costs_accum += attn_val * dt
        self.escalation_levels.append(float(tier))

        self.attention_response_window = max(0.0, self.attention_response_window - dt)
        # Count signal events, not every frame of a sustained gesture. Require
        # a quiet interval/cooldown before another attempt can be attributed.
        if (target_is_human and tier >= 2 and self.previous_tier < 2 and
                self.sim_time - self.last_attention_time >= 3.0):
            self.attention_attempts += 1
            self.attention_response_window = 2.0
            self.last_attention_time = self.sim_time
        if human_responded and target_is_human and self.attention_response_window > 0:
            self.attention_successes += 1
            self.attention_response_window = 0.0
        self.previous_tier = tier

        if is_spam:
            self.signal_spam_steps += 1

        return self.compute_metrics()

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Tính toán bộ chỉ số KPI bắt buộc theo Doc 13 Section 13.
        """
        n_steps = max(1, self.step_idx)

        # S0 Metrics
        motion_jerk = self.jerk_accum / n_steps
        stationary_ratio = self.stillness_steps / n_steps
        servo_saturation_rate = self.saturation_steps / n_steps
        safety_override_rate = self.total_overrides / n_steps

        # S1 Metrics
        pan_vals = [p for p, t in self.view_angles]
        pan_coverage = (max(pan_vals) - min(pan_vals)) / 120.0 if len(pan_vals) > 0 else 0.0
        pan_coverage = float(np.clip(pan_coverage, 0.0, 1.0))
        view_diversity = float(np.clip(np.std(pan_vals) / 30.0, 0.0, 1.0)) if len(pan_vals) > 1 else 0.0

        # S2 & S5 Metrics
        stare_rate = self.stare_violations / n_steps
        mean_gaze = self.human_gaze_duration

        # S3 Metrics
        handoff_rate = self.tracking_handoff_count / max(1, self.tracking_handoff_count + 1)
        tracking_timeout_rate = self.tracking_timeouts / max(1, self.tracking_handoff_count)

        # S4 Metrics
        acoustic_response_rate = (self.acoustic_responses / max(1, self.acoustic_event_count)) if self.acoustic_event_count > 0 else 1.0

        # S6 Metrics
        attention_success_rate = self.attention_successes / max(1, self.attention_attempts)
        signal_spam_rate = self.signal_spam_steps / n_steps
        mean_escalation = float(np.mean(self.escalation_levels)) if len(self.escalation_levels) > 0 else 0.0

        return {
            "scenario": self.scenario.value,
            "motion_jerk": motion_jerk,
            "stationary_ratio": stationary_ratio,
            "servo_saturation_rate": servo_saturation_rate,
            "safety_override_rate": safety_override_rate,
            "pan_coverage": pan_coverage,
            "view_diversity": view_diversity,
            "object_inspections": self.object_inspections,
            "human_orientations": self.human_orientations,
            "stare_rate": stare_rate,
            "tracking_handoffs": self.tracking_handoff_count,
            "tracking_timeout_rate": tracking_timeout_rate,
            "acoustic_events": self.acoustic_event_count,
            "acoustic_responses": self.acoustic_responses,
            "acoustic_response_rate": acoustic_response_rate,
            "attention_success_rate": attention_success_rate,
            "attention_attempts": self.attention_attempts,
            "attention_successes": self.attention_successes,
            "tracking_motion_steps": self.tracking_motion_steps,
            "tracking_aligned_ratio": self.tracking_aligned_steps / max(1, self.tracking_motion_steps),
            "tracking_mean_error_rad": self.tracking_error_sum / max(1, self.tracking_motion_steps),
            "collision_steps": self.collision_steps,
            "unstable_steps": self.unstable_steps,
            "signal_spam_rate": signal_spam_rate,
            "mean_escalation_level": mean_escalation,
            "is_tracking_active": self.tracking_status.active
        }

    def evaluate_acceptance(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Kiểm tra Acceptance Criteria của từng Scenario theo chuẩn nghiệm thu khắt khe Doc 13 Section 5 - 12.
        """
        sc = self.scenario
        passed = True
        reasons = []
        required = {
            "S0": ("stationary_ratio", "safety_override_rate"),
            "S1": ("pan_coverage", "view_diversity"),
            "S2": ("stare_rate", "object_inspections"),
            "S3": ("tracking_handoffs", "tracking_timeout_rate", "tracking_motion_steps", "tracking_aligned_ratio"),
            "S4": ("acoustic_responses",), "S5": ("human_orientations", "stare_rate"),
            "S6": ("signal_spam_rate", "attention_success_rate", "attention_attempts"),
            "S7": ("safety_override_rate",),
        }
        for metric in ("motion_jerk", "servo_saturation_rate", "collision_steps", "unstable_steps") + required[sc.value]:
            if metric not in metrics or not np.isfinite(metrics[metric]):
                passed = False
                reasons.append(f"Missing/nonfinite metric: {metric}")
        if not passed:
            return False, reasons
        if metrics.get("collision_steps", 0) > 0 or metrics.get("unstable_steps", 0) > 0:
            passed = False
            reasons.append("Collision or unstable simulation")
        if metrics.get("motion_jerk", float("inf")) > .05:
            passed = False
            reasons.append("Command variation exceeds 0.05")
        if metrics.get("servo_saturation_rate", float("inf")) > .15:
            passed = False
            reasons.append("Servo saturation exceeds 15%")

        if sc == BehavioralScenario.S0_MOTOR_SANITY:
            # Doc 13 Section 5.8: Jerk thấp, không saturation, giữ stillness, override == 0
            if metrics["motion_jerk"] > 0.05:
                passed = False
                reasons.append(f"Jerk quá cao ({metrics['motion_jerk']:.4f} > 0.05)")
            if metrics["servo_saturation_rate"] > 0.15:
                passed = False
                reasons.append(f"Servo saturation quá cao ({metrics['servo_saturation_rate']:.2%} > 15%)")
            if metrics["stationary_ratio"] < 0.50:
                passed = False
                reasons.append(f"Chưa đạt tỷ lệ nghỉ nghiêm ngặt ({metrics['stationary_ratio']:.2%} < 50%)")
            if metrics["safety_override_rate"] > 0.05:
                passed = False
                reasons.append(f"Safety override loop ({metrics['safety_override_rate']:.2%})")

        elif sc == BehavioralScenario.S1_VISUAL_EXPLORATION:
            # Doc 13 Section 6.8: Độ bao phủ pan, độ đa dạng góc nhìn, không va giới hạn
            if metrics["pan_coverage"] < 0.20:
                passed = False
                reasons.append(f"Độ bao phủ góc nhìn hẹp ({metrics['pan_coverage']:.2%} < 20%)")
            if metrics["view_diversity"] < 0.15:
                passed = False
                reasons.append(f"Độ đa dạng góc nhìn thấp ({metrics['view_diversity']:.3f} < 0.15)")
            if metrics["servo_saturation_rate"] > 0.20:
                passed = False
                reasons.append(f"Servo saturation cao khi scan ({metrics['servo_saturation_rate']:.2%} > 20%)")

        elif sc == BehavioralScenario.S2_OBJECT_CURIOSITY:
            # Doc 13 Section 7.8: Inspect vật thể, novelty decay, anti-stare
            if metrics["stare_rate"] > 0.15:
                passed = False
                reasons.append(f"Hiện tượng nhìn chằm chằm Stare Forever ({metrics['stare_rate']:.2%} > 15%)")
            if metrics["object_inspections"] < 1:
                passed = False
                reasons.append("Không thực hiện quan sát vật thể mới")

        elif sc == BehavioralScenario.S3_DYNAMIC_OBJECT:
            # Doc 13 Section 8.8: Dynamic motion -> TrackingRequest -> smooth handoff & release
            if metrics["tracking_handoffs"] < 1:
                passed = False
                reasons.append("Không kích hoạt TrackingRequest handoff khi phát hiện chuyển động")
            if metrics["tracking_timeout_rate"] > 0.25:
                passed = False
                reasons.append(f"Tracking policy timeout quá cao ({metrics['tracking_timeout_rate']:.2%} > 25%)")
            if metrics.get("tracking_motion_steps", 0) < 1 or metrics.get("tracking_aligned_ratio", 0) < .60:
                passed = False
                reasons.append("Moving-target visual alignment below 60%")

        elif sc == BehavioralScenario.S4_ACOUSTIC_EVENT:
            # Doc 13 Section 9.8: Phản ứng âm thanh, orient nhanh, refractory cooldown
            if metrics.get("acoustic_responses", 0) < 1:
                passed = False
                reasons.append("Không có phản ứng định hướng nguồn âm thanh")
            if metrics["motion_jerk"] > 0.05:
                passed = False
                reasons.append(f"Chuyển động giật cục khi quay theo âm thanh ({metrics['motion_jerk']:.4f} > 0.05)")

        elif sc == BehavioralScenario.S5_HUMAN_PRESENCE:
            # Doc 13 Section 10.8: Human gaze, mutual attention, look-away, stare bound
            if metrics["human_orientations"] < 1:
                passed = False
                reasons.append("Không hướng mắt tương tác khi có người xuất hiện")
            if metrics["stare_rate"] > 0.40:
                passed = False
                reasons.append(f"Tỷ lệ nhìn chằm chằm người quá cao ({metrics['stare_rate']:.2%} > 40%)")

        elif sc == BehavioralScenario.S6_ATTENTION_SEEKING:
            # Doc 13 Section 11.8: 5-tier escalation, anti-spam, success rate
            if metrics["signal_spam_rate"] > 0.05:
                passed = False
                reasons.append(f"Spam tín hiệu âm thanh / biểu cảm ({metrics['signal_spam_rate']:.2%} > 5%)")
            if metrics["attention_success_rate"] < 0.70:
                passed = False
                reasons.append(f"Hiệu quả thu hút chú ý thấp ({metrics['attention_success_rate']:.2%} < 70%)")
            if metrics.get("attention_attempts", 0) < 1:
                passed = False
                reasons.append("No actual attention attempt")

        elif sc == BehavioralScenario.S7_MIXED:
            # Doc 13 Section 12.8: Toàn bộ arbitration, safety override thấp, motion smooth
            if metrics["safety_override_rate"] > 0.10:
                passed = False
                reasons.append(f"Safety override xảy ra quá thường xuyên ({metrics['safety_override_rate']:.2%} > 10%)")
            if metrics["motion_jerk"] > 0.05:
                passed = False
                reasons.append(f"Jerk tổng thể trong môi trường hỗn hợp cao ({metrics['motion_jerk']:.4f} > 0.05)")

        return passed, reasons
