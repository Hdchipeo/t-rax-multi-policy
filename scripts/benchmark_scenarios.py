#!/usr/bin/env python3
"""
T-RAX Autonomous Behavioral Dynamics: Scenario Benchmark Suite (Doc 13 - Behavioral Scenarios Implementation Plan)
Thực hiện đánh giá toàn diện Ma trận kịch bản từ S0 đến S7:
  - S0: Motor Sanity (Stability, stillness, jerk, limit saturation)
  - S1: Visual Exploration (Scan, view signature diversity, coverage)
  - S2: Object Curiosity (Static object inspection, novelty decay, anti-stare)
  - S3: Dynamic Object (Ball rolling, motion detection, TrackingRequest handoff)
  - S4: Acoustic Event (Sound transients, orientation, refractory cooldown)
  - S5: Human Presence (Human gaze, mutual attention, look-away, anti-stare)
  - S6: Attention Seeking (5-tier intensity escalation, anti-spam, cost balance)
  - S7: Mixed Environment (Full multi-event integration & runtime arbitration)
"""

import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv
from trax_rl.dynamics.scenario_manager import BehavioralScenario
from trax_rl.evaluation import evaluate_scenarios

SCENARIOS = [
    BehavioralScenario.S0_MOTOR_SANITY,
    BehavioralScenario.S1_VISUAL_EXPLORATION,
    BehavioralScenario.S2_OBJECT_CURIOSITY,
    BehavioralScenario.S3_DYNAMIC_OBJECT,
    BehavioralScenario.S4_ACOUSTIC_EVENT,
    BehavioralScenario.S5_HUMAN_PRESENCE,
    BehavioralScenario.S6_ATTENTION_SEEKING,
    BehavioralScenario.S7_MIXED,
]

def run_single_scenario(model, scenario: BehavioralScenario, seeds: list, steps_per_seed: int = 200) -> dict:
    result = evaluate_scenarios(model, env_options={"enable_domain_randomization":False},
        seeds=seeds, steps=steps_per_seed, scenarios=[scenario.value])
    metrics_history = [r["metrics"] for r in result["episodes"]]
    agg_metrics = {key:float(np.mean([m[key] for m in metrics_history]))
        for key, value in metrics_history[0].items() if isinstance(value, (int, float))}
    agg_metrics.update(scenario=scenario.value, mean_reward=result["mean_reward"],
        passed=result["passed"], reasons=sorted(set(reason for r in result["episodes"] for reason in r["reasons"])))
    return agg_metrics

def main():
    parser = argparse.ArgumentParser(description="T-RAX Doc 13 Scenario Benchmark Suite")
    parser.add_argument("--model", type=str, default="checkpoints/autonomous_behavioral_dynamics/final_model.zip")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="3 seeds required by Doc 13 Section 14")
    parser.add_argument("--steps", type=int, default=200, help="Steps per scenario episode")
    args = parser.parse_args()

    print("=========================================================================================")
    print("🏆 T-RAX AUTONOMOUS BEHAVIORAL SCENARIO BENCHMARK (DOC 13)")
    print(f"📦 Checkpoint  : {args.model}")
    print(f"🌱 Random Seeds: {args.seeds} (Multi-seed statistical validation)")
    print(f"⏱️ Steps/Seed  : {args.steps}")
    print("=========================================================================================\n")

    model = None
    if os.path.exists(args.model):
        print(f"✅ Loaded trained model from {args.model}")
        try:
            model = PPO.load(args.model, device="cpu")
        except Exception:
            model = SAC.load(args.model, device="cpu")
    else:
        raise FileNotFoundError(args.model)

    scenarios_to_run = SCENARIOS if args.scenario == "all" else [BehavioralScenario(args.scenario)]

    results = []
    all_passed = True

    for sc in scenarios_to_run:
        print(f"Running Scenario {sc.value} across {len(args.seeds)} seeds ...", end="", flush=True)
        res = run_single_scenario(model, sc, args.seeds, steps_per_seed=args.steps)
        results.append(res)
        status_str = "✅ PASSED" if res["passed"] else f"❌ FAILED ({', '.join(res['reasons'])})"
        print(f" -> {status_str}")
        if not res["passed"]:
            all_passed = False

    # Hiển thị bảng tổng hợp Markdown
    print("\n=========================================================================================")
    print("📊 BẢNG TỔNG HỢP MA TRẬN KỊCH BẢN (DOC 13 SCENARIO TEST MATRIX)")
    print("=========================================================================================")
    headers = ["Scenario", "Reward", "Jerk", "Saturation", "Override", "Coverage", "Stare Rate", "Handoff", "Spam", "Status"]
    row_fmt = "| {:<8} | {:>7.1f} | {:>6.4f} | {:>10.2%} | {:>8.2%} | {:>8.2%} | {:>10.2%} | {:>7.1f} | {:>6.2%} | {:<8} |"
    sep_line = "|----------|---------|--------|------------|----------|----------|------------|---------|--------|----------|"

    print("| " + " | ".join(headers) + " |")
    print(sep_line)
    for r in results:
        status_icon = "PASSED" if r["passed"] else "FAIL"
        print(row_fmt.format(
            r["scenario"],
            r["mean_reward"],
            r["motion_jerk"],
            r["servo_saturation_rate"],
            r["safety_override_rate"],
            r["pan_coverage"],
            r["stare_rate"],
            r["tracking_handoffs"],
            r["signal_spam_rate"],
            status_icon
        ))

    print("\n=========================================================================================")
    if all_passed:
        print("🎉 TẤT CẢ CÁC KỊCH BẢN TỪ S0 ĐẾN S7 ĐẠU TIÊU CHUẨN CHẤP THUẬN (ACCEPTANCE CRITERIA)!")
        print("Simulation gates passed; hardware validation is still required.")
    else:
        print("⚠️ Một số kịch bản cần tiếp tục tinh chỉnh trước khi chuyển giao phần cứng.")
    print("=========================================================================================\n")
    return 0 if all_passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
