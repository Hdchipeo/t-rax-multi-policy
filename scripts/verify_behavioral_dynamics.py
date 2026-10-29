#!/usr/bin/env python3
"""
T-RAX Autonomous Behavioral Dynamics: Comprehensive Verification Suite
Kiểm thử toàn diện 12 tài liệu thiết kế (Doc 01 - Doc 12):
1. Khởi tạo TraxBehavioralEnv & xác thực Observation Space 39-dim
2. Kiểm tra EpisodicNoveltyEngine (Episodic Memory, Anti-stare Decay)
3. Kiểm tra HumanAgent & SocialAttentionDynamics (Tiers 0..4, Anti-spam, Handoff)
4. Kiểm tra InternalDrivesEngine (Curiosity, Boredom, Social, Energy, Confidence)
5. Kiểm tra EventScheduler (Poisson stimuli)
6. Kiểm tra Deterministic Safety Layer (Clamp, Limit, Safe override)
7. Chạy 100 bước mô phỏng thực tế và xuất bảng chỉ số tổng hợp
"""

import os
import sys
import numpy as np

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv
from trax_rl.dynamics.curiosity_engine import EpisodicNoveltyEngine
from trax_rl.dynamics.social_dynamics import HumanAgent, SocialAttentionDynamics
from trax_rl.dynamics.internal_drives import InternalDrivesEngine
from trax_rl.dynamics.behavior_manager import BehaviorManager, Behavior

def run_behavioral_dynamics_tests():
    print("==================================================================")
    print("🧪 T-RAX AUTONOMOUS BEHAVIORAL DYNAMICS VERIFICATION SUITE")
    print("==================================================================")

    # -------------------------------------------------------------
    # TEST 1: Internal Drives Engine
    # -------------------------------------------------------------
    print("\n[TEST 1/6] Testing Internal Drives Engine ...")
    drives = InternalDrivesEngine()
    drives.reset()
    init_vec = drives.get_vector()
    assert init_vec.shape == (5,), f"Expected shape (5,), got {init_vec.shape}"
    assert np.all((init_vec >= 0.0) & (init_vec <= 1.0)), "Drives out of [0, 1] bounds!"

    # Simulate 10 steps of high novelty & human interaction
    for _ in range(10):
        drives.step(dt=0.1, novelty_score=0.8, human_visible=True, human_engaged=True, action_energy=0.2, is_resting=False)
    vec_after = drives.get_vector()
    assert vec_after[1] < init_vec[1], "Boredom should decrease during interaction!"
    assert vec_after[2] < init_vec[2], "Social need should decrease when engaged!"
    print(f"✅ Internal Drives verified! Initial: {init_vec}, After interaction: {vec_after}")

    # -------------------------------------------------------------
    # TEST 2: Episodic Novelty & Anti-Stare Decay
    # -------------------------------------------------------------
    print("\n[TEST 2/6] Testing Episodic Novelty Engine & Gaze Decay ...")
    novelty_engine = EpisodicNoveltyEngine(feature_dim=6, max_memory_size=100)
    novelty_engine.reset()

    state1 = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.2], dtype=np.float32)
    nov1, view1, obj1 = novelty_engine.evaluate(state1, dt=0.1)
    assert nov1 > 0.8, f"First state should have high novelty! Got: {nov1}"

    # Stare at same state for 15 steps -> Novelty should decay
    decayed_nov = nov1
    for _ in range(15):
        decayed_nov, _, _ = novelty_engine.evaluate(state1, dt=0.1)
    assert decayed_nov < nov1 * 0.5, f"Anti-stare decay failed! nov1={nov1}, decayed_nov={decayed_nov}"

    # Move to completely new state -> Novelty should spike again
    state2 = np.array([0.9, -0.8, 0.0, 0.9, 0.1, 0.9], dtype=np.float32)
    nov2, view2, obj2 = novelty_engine.evaluate(state2, dt=0.1)
    assert nov2 > decayed_nov, f"New state should have higher novelty than decayed state! nov2={nov2}"
    print(f"✅ Episodic Novelty verified! First: {nov1:.3f} -> Stare Decayed: {decayed_nov:.3f} -> New View Spike: {nov2:.3f}")

    # -------------------------------------------------------------
    # TEST 3: Social Attention & Human Agent
    # -------------------------------------------------------------
    print("\n[TEST 3/6] Testing Social Attention Dynamics & Expression Tiers ...")
    social = SocialAttentionDynamics(audio_cooldown_sec=2.0)
    social.reset()
    human = HumanAgent()
    human.reset(is_present=True)

    tier0, _, emit0, spam0 = social.decode_attention_intensity(0.10, dt=0.1)
    assert tier0 == 0 and not emit0, "Intensity 0.1 should be Tier 0 (Quiet)"

    tier2, gest2, _, _ = social.decode_attention_intensity(0.50, dt=0.1)
    assert tier2 == 2 and gest2, "Intensity 0.5 should trigger Tier 2 with head gesture"

    tier4, _, emit4, _ = social.decode_attention_intensity(0.95, dt=0.1)
    assert tier4 == 4 and emit4, "First Tier 4 signal should emit audio"

    # Immediately try to trigger audio again -> Anti-spam should catch it
    tier4_spam, _, emit_spam, is_spam = social.decode_attention_intensity(0.95, dt=0.1)
    assert is_spam and not emit_spam, "Consecutive audio emission must be blocked as spam!"

    print(f"✅ Social Attention verified! Tier 0: OK, Tier 2 (Gesture): OK, Tier 4 (Audio): OK, Anti-Spam Shield: OK")

    # -------------------------------------------------------------
    # TEST 4: Behavior Manager Arbiter & Deterministic Safety Layer
    # -------------------------------------------------------------
    print("\n[TEST 4/6] Testing Behavior Manager Arbiter & Safety Layer ...")
    bm = BehaviorManager()
    bm.reset()

    # Priority 1: Emergency Stop
    act_beh = bm.arbitrate(dt=0.1, emergency_stop=True, user_command_active=False,
                           obstacle_dist_m=1.0, human_engagement=0.9, ball_confidence=0.9, target_visible=True)
    assert act_beh == Behavior.SAFE_STOP, f"Emergency stop priority failed! Got: {act_beh}"

    # Priority 2: User Command
    act_beh = bm.arbitrate(dt=0.1, emergency_stop=False, user_command_active=True,
                           obstacle_dist_m=1.0, human_engagement=0.9, ball_confidence=0.9, target_visible=True)
    assert act_beh == Behavior.COMMAND, f"User command priority failed! Got: {act_beh}"

    # Safety Layer clamping: Excessive velocity and servo violation
    safe_act, overridden, reason = bm.sanitize_action(
        linear_vel=10.0, angular_vel=5.0, pan_vel=-5.0, tilt_vel=0.0, attention_intensity=2.0,
        current_pan_rad=-1.047, current_tilt_rad=0.0, front_obstacle_m=0.10
    )
    assert safe_act[0] == 0.0, "Obstacle near 0.10m must force linear velocity to 0!"
    assert safe_act[2] == 0.0, "Pan velocity toward min limit must be clamped to 0!"
    assert safe_act[4] <= 1.0, "Attention intensity must be clipped to [0, 1]!"

    # Blind Reverse Prevention Clamping
    safe_rev, over_rev, reason_rev = bm.sanitize_action(
        linear_vel=-0.35, angular_vel=0.0, pan_vel=0.0, tilt_vel=0.0, attention_intensity=0.0,
        current_pan_rad=0.0, current_tilt_rad=0.0, front_obstacle_m=1.5, target_dist=1.5
    )
    assert safe_rev[0] == 0.0, "Unprovoked blind reverse must be clamped to 0!"
    assert reason_rev == "BLIND_REVERSE_CLAMP", f"Expected BLIND_REVERSE_CLAMP, got {reason_rev}"

    # Permitted reverse for obstacle clearance
    safe_esc, _, _ = bm.sanitize_action(
        linear_vel=-0.35, angular_vel=0.0, pan_vel=0.0, tilt_vel=0.0, attention_intensity=0.0,
        current_pan_rad=0.0, current_tilt_rad=0.0, front_obstacle_m=0.15, target_dist=1.5
    )
    assert safe_esc[0] <= -0.15, f"Reverse must be permitted when front obstacle is critically close! Got: {safe_esc[0]}"

    # Permitted reverse for social space restoration
    safe_soc, _, _ = bm.sanitize_action(
        linear_vel=-0.35, angular_vel=0.0, pan_vel=0.0, tilt_vel=0.0, attention_intensity=0.0,
        current_pan_rad=0.0, current_tilt_rad=0.0, front_obstacle_m=1.5, target_dist=0.35
    )
    assert safe_soc[0] <= -0.15, f"Reverse must be permitted when human is crowding intimate space! Got: {safe_soc[0]}"

    # RC1: Bóng ở gần (< 1.20m) KHÔNG được mở khóa lùi mù (personal space chỉ dành cho người)
    bm.last_safe_lin = 0.0
    safe_ball, _, reason_ball = bm.sanitize_action(
        linear_vel=-0.35, angular_vel=0.0, pan_vel=0.0, tilt_vel=0.0, attention_intensity=0.0,
        current_pan_rad=0.0, current_tilt_rad=0.0, front_obstacle_m=1.5, target_dist=0.50,
        target_is_human=False
    )
    assert safe_ball[0] == 0.0 and reason_ball == "BLIND_REVERSE_CLAMP", f"Ball target must not unlock blind reverse! Got: {safe_ball[0]}, {reason_ball}"

    # RC1b: Lệnh lùi chậm trong khe [-0.06, -0.03) phải bị chặn (không còn lọt qua deadband)
    bm.last_safe_lin = 0.0
    safe_creep, _, _ = bm.sanitize_action(
        linear_vel=-0.045, angular_vel=0.0, pan_vel=0.0, tilt_vel=0.0, attention_intensity=0.0,
        current_pan_rad=0.0, current_tilt_rad=0.0, front_obstacle_m=1.5, target_dist=1.5
    )
    assert safe_creep[0] == 0.0, f"Slow creep reverse must be clamped! Got: {safe_creep[0]}"

    # Dynamic Target Prioritization: social_need < 0.20 -> Ball priority; social_need >= 0.20 -> Human priority
    bm.active_time_sec = 5.0
    bm.cooldown_timer_sec = 0.0
    b_soc = bm.arbitrate(dt=0.1, human_engagement=0.85, ball_confidence=0.88, target_visible=True, target_dist=1.2, social_need=0.50)
    assert b_soc == Behavior.SOCIAL, f"Expected SOCIAL, got {b_soc}"

    bm.active_time_sec = 5.0
    bm.cooldown_timer_sec = 0.0
    b_ball = bm.arbitrate(dt=0.1, human_engagement=0.85, ball_confidence=0.88, target_visible=True, target_dist=1.2, social_need=0.15)
    assert b_ball in (Behavior.APPROACH, Behavior.BALL), f"Expected APPROACH/BALL, got {b_ball}"
    assert bm.behavior_trigger == "BOREDOM_BALL_PRIORITY", f"Expected BOREDOM_BALL_PRIORITY, got {bm.behavior_trigger}"

    print(f"✅ Behavior Manager & Safety Layer verified! Emergency: OK, Manual: OK, Hard Clamp: OK, Blind Reverse Clamp: OK, Target Prioritization: OK")

    # -------------------------------------------------------------
    # TEST 5: TraxBehavioralEnv 39-Dim Observation & Step Integration
    # -------------------------------------------------------------
    print("\n[TEST 5/6] Testing Full TraxBehavioralEnv (Continuous 5D Action, 39D Observation) ...")
    env = TraxBehavioralEnv(action_mode="continuous")
    obs, info = env.reset(seed=123)
    assert obs.shape == (39,), f"Observation shape mismatch! Expected (39,), got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf!"

    total_r = 0.0
    for step in range(50):
        # Continuous action: [lin, ang, pan_vel, tilt_vel, attn]
        test_action = np.array([0.1, 0.0, 0.3 * np.sin(step * 0.2), 0.0, 0.4], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(test_action)
        total_r += reward

        assert obs.shape == (39,), f"Step {step}: Observation corrupted!"
        assert np.isfinite(reward), f"Step {step}: Reward is NaN or Inf!"

    # Verify new Upgrade Plan components in info
    assert "canonical_reward" in info, "Missing canonical_reward in info!"
    for key in ("novelty", "exploration", "social", "quiet", "energy", "smoothness", "safety", "total"):
        assert key in info["canonical_reward"], f"Missing key {key} in canonical_reward!"

    assert "clearances" in info, "Missing multi-sector clearances in info!"
    for sector in ("front", "front_left", "front_right", "left", "right", "rear"):
        assert sector in info["clearances"], f"Missing sector {sector} in clearances!"
        assert 0.0 <= info["clearances"][sector] <= 2.5, f"Clearance for {sector} out of range!"

    assert "pose" in info, "Missing pose in info!"
    assert -180.0 <= info["pose"]["yaw_deg"] <= 180.0, f"Wrapped yaw_deg out of [-180, 180] range: {info['pose']['yaw_deg']}"

    assert "safety_metrics" in info, "Missing safety_metrics in info!"
    assert "trigger" in info["safety_metrics"], "Missing trigger in safety_metrics!"
    assert "active_policy" in info["safety_metrics"], "Missing active_policy in safety_metrics!"

    print(f"✅ Full Behavioral Env 50-step run passed! Total Reward: {total_r:.2f}")
    print(f"   Canonical reward breakdown: {info['canonical_reward']}")
    print(f"   Multi-sector clearances: {info['clearances']}")
    print(f"   Wrapped pose: {info['pose']}")
    print(f"   Internal drives snapshot: {info['internal_drives']}")

    # -------------------------------------------------------------
    # TEST 6: Discrete Macro-Action Mode (Firmware Compatibility)
    # -------------------------------------------------------------
    print("\n[TEST 6/7] Testing Discrete Mode (Firmware Primitive Actions Compatibility) ...")
    env_discrete = TraxBehavioralEnv(action_mode="discrete")
    obs_d, _ = env_discrete.reset(seed=456)
    for act in range(5):
        obs_d, r, _, _, _ = env_discrete.step(act)
        assert obs_d.shape == (39,), f"Discrete action {act} returned invalid obs!"

    print("✅ Discrete Macro-Action mode verified successfully!")

    # -------------------------------------------------------------
    # TEST 7: Robot-Centric Observation Schema & Dynamic Fear Decay
    # -------------------------------------------------------------
    print("\n[TEST 7/7] Testing Robot-Centric Observation Schema & Dynamic Fear Decay ...")
    from trax_rl.observations.schemas_v2 import build_robot_centric_observation
    robot_obs = build_robot_centric_observation(
        linear_velocity=0.25,
        angular_velocity=-0.40,
        robot_yaw_rad=-11.2, # Arbitrary accumulated yaw (~ -641 deg)
        pan_norm=0.15,
        tilt_norm=-0.10,
        target_visible=True,
        target_distance=0.85,
        target_bearing_rad=0.35,
        front_tof_distance=1.45,
        drives=info["internal_drives"],
        prev_action=np.array([0.2, -0.3, 0.0, 0.0, 0.2], dtype=np.float32)
    )
    assert robot_obs["features"].shape == (18,), f"Expected 18-dim robot-centric features, got {robot_obs['features'].shape}"
    assert -np.pi <= robot_obs["wrapped_yaw_rad"] <= np.pi, "Wrapped yaw rad out of [-pi, pi]!"
    assert np.isclose(robot_obs["sin_yaw"]**2 + robot_obs["cos_yaw"]**2, 1.0, atol=1e-5), "sin^2 + cos^2 != 1.0!"

    # Verify Multi-Policy Architecture (Upgrade Plan Phase 3 & 4)
    bm = env.behavior_manager
    action_explore = bm.policy_registry[Behavior.EXPLORE].act(obs, info)
    assert action_explore.shape == (5,), "Explore policy action shape mismatch!"
    action_look = bm.policy_registry[Behavior.LOOK_AROUND].act(obs, info)
    assert action_look[0] == 0.0 and action_look[1] == 0.0, "LookAround policy must keep chassis static!"
    action_approach = bm.policy_registry[Behavior.APPROACH].act(obs, info)
    assert action_approach.shape == (5,), "Approach policy action shape mismatch!"
    action_escape = bm.policy_registry[Behavior.ESCAPE].act(obs, info)
    assert action_escape.shape == (5,), "Escape policy action shape mismatch!"
    print(f"✅ Multi-Policy Framework verified! All 4 specialized sub-policies executed successfully.")

    # Verify Dynamic Fear Decay: fear spikes under threat, then decays when safe (Upgrade Plan Section 3.1)
    drives_engine = InternalDrivesEngine()
    drives_engine.reset()
    assert drives_engine.fear == 0.0, "Initial fear must be 0!"

    # Threat event: fear increases
    for _ in range(5):
        drives_engine.step(dt=0.1, novelty_score=0.1, human_visible=False, human_engaged=False,
                           action_energy=0.1, is_resting=False, threat_score=0.8)
    spiked_fear = drives_engine.fear
    assert spiked_fear > 0.4, f"Fear should spike under threat! Got: {spiked_fear}"

    # Safe environment: fear decays exponentially
    for _ in range(20):
        drives_engine.step(dt=0.1, novelty_score=0.1, human_visible=False, human_engaged=False,
                           action_energy=0.0, is_resting=True, threat_score=0.0)
    decayed_fear = drives_engine.fear
    assert decayed_fear < spiked_fear * 0.35, f"Fear must decay when safe! spiked={spiked_fear}, decayed={decayed_fear}"
    print(f"✅ Dynamic Fear verified! Spiked under threat: {spiked_fear:.3f} -> Decayed when safe: {decayed_fear:.3f}")
    print(f"✅ Robot-Centric Observation verified! Feature vector: shape {robot_obs['features'].shape}")

    print("\n==================================================================")
    print("🎉 ALL 7 COMPREHENSIVE BEHAVIORAL DYNAMICS & UPGRADE TESTS PASSED!")
    print("==================================================================")

if __name__ == "__main__":
    run_behavioral_dynamics_tests()
