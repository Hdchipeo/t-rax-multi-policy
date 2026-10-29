#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_idle_env import TraxIdleEnv

def main():
    parser = argparse.ArgumentParser(description="Train T-RAX RL Policy on Mac M4")
    parser.add_argument("--config", type=str, default="configs/policies/idle_v1.yaml", help="Path to config yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    policy_name = cfg["policy"]["name"]
    total_timesteps = args.total_timesteps or cfg["training"]["total_timesteps"]
    n_envs = cfg["training"].get("n_envs", 8)
    lr = cfg["training"].get("learning_rate", 3e-4)
    net_arch = cfg["network"].get("net_arch", [64, 64])

    print(f"=== Training T-RAX RL Policy: {policy_name} ===")
    print(f"Algorithm: PPO | Steps: {total_timesteps:,} | Envs: {n_envs} | Seed: {args.seed}")
    print(f"Network Architecture: {net_arch} | Learning Rate: {lr}")

    # Tạo thư mục runs & checkpoints
    os.makedirs(f"runs/{policy_name}", exist_ok=True)
    os.makedirs(f"checkpoints/{policy_name}", exist_ok=True)

    # Khởi tạo vectorized environments
    env = make_vec_env(TraxIdleEnv, n_envs=n_envs, seed=args.seed)

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=cfg["training"].get("n_steps", 2048),
        batch_size=cfg["training"].get("batch_size", 64),
        n_epochs=cfg["training"].get("n_epochs", 10),
        gamma=cfg["training"].get("gamma", 0.99),
        gae_lambda=cfg["training"].get("gae_lambda", 0.95),
        clip_range=cfg["training"].get("clip_range", 0.2),
        ent_coef=cfg["training"].get("ent_coef", 0.01),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{policy_name}",
        device="cpu"  # CPU is faster for small MLPs on Apple Silicon
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1000),
        save_path=f"checkpoints/{policy_name}",
        name_prefix="model_checkpoint"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    final_path = f"checkpoints/{policy_name}/final_model"
    model.save(final_path)
    print(f"Training completed successfully! Saved final model to: {final_path}.zip")

if __name__ == "__main__":
    main()
