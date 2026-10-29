import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_idle_env import TraxIdleEnv

EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]

def evaluate_policy(model_path: str, episodes_per_seed: int = 2):
    print(f"Loading model for deterministic evaluation: {model_path}")
    model = PPO.load(model_path, device="cpu")
    env = TraxIdleEnv()

    total_rewards = []
    success_counts = []
    repetition_rates = []
    rest_rates = []

    print(f"Running deterministic test scenarios across {len(EVAL_SEEDS)} fixed seeds...")

    for seed in EVAL_SEEDS:
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=seed + ep * 10)
            ep_reward = 0.0
            actions = []
            step_infos = []
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                step_infos.append(info)
                ep_reward += reward
                steps += 1
                if terminated or truncated:
                    break

            actions = np.array(actions)
            total_rewards.append(ep_reward)

            # Đánh giá tracking success: Tỷ lệ các bước track thành công khi mục tiêu hiển thị
            visible_mask = np.array([step_info.get("visible", False) for step_info in step_infos])
            tracked_mask = np.array([step_info.get("tracked", False) for step_info in step_infos])
            if np.sum(visible_mask) > 0:
                success_ratio = np.sum(tracked_mask & visible_mask) / np.sum(visible_mask)
            else:
                # Nếu phòng không có người, tỷ lệ REST đạt chuẩn là thành công
                success_ratio = 1.0 if np.mean(actions == 0) > 0.8 else 0.0
            success_counts.append(success_ratio)
            
            # Tính repetition rate của các hành vi chủ động (không tính REST)
            non_rest_mask = (actions[1:] != 0) & (actions[:-1] != 0)
            if np.sum(non_rest_mask) > 0:
                rep_count = np.sum((actions[1:] == actions[:-1]) & non_rest_mask)
                repetition_rates.append(rep_count / np.sum(non_rest_mask))
            else:
                repetition_rates.append(0.0)
            
            # Tính tỷ lệ REST (action 0)
            rest_rates.append(np.mean(actions == 0))

    mean_reward = np.mean(total_rewards)
    mean_success = np.mean(success_counts)
    mean_rep = np.mean(repetition_rates)
    mean_rest = np.mean(rest_rates)

    print("\n================ EVALUATION REPORT ================")
    print(f"Episodes Tested       : {len(total_rewards)}")
    print(f"Mean Episode Reward   : {mean_reward:.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Tracking Success Rate : {mean_success * 100:.1f}%")
    print(f"Repetition Rate       : {mean_rep * 100:.1f}%")
    print(f"REST Action Ratio     : {mean_rest * 100:.1f}%")
    print("====================================================")

    # Đánh giá tiêu chuẩn Release Gate (Definition of Done)
    passed_gate = (mean_success >= 0.70 and mean_rep <= 0.60)
    print(f"Promotion Gate Status : {'PASSED (Ready for Export)' if passed_gate else 'NEEDS RETRAINING'}")
    return passed_gate

def main():
    parser = argparse.ArgumentParser(description="Evaluate T-RAX Autonomous Idle Policy")
    parser.add_argument("--model", type=str, default="checkpoints/autonomous_idle/final_model.zip", help="Path to SB3 zip")
    args = parser.parse_args()

    evaluate_policy(args.model)

if __name__ == "__main__":
    main()
