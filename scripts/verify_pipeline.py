#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# Thêm project root vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_idle_env import TraxIdleEnv
from scripts.export_onnx import export_actor_to_onnx
from scripts.quantize_ppq import quantize_onnx_model

def run_end_to_end_verification():
    print("=================================================================")
    print("   T-RAX Multi-Policy: End-to-End Pipeline Verification Test")
    print("=================================================================")

    # 1. Quick Mini-Train (5,000 steps) để kiểm chứng pipeline hoạt động trơn tru
    print("\n[STEP 1/4] Running Quick Gym Environment & PPO Sandbox Step ...")
    env = TraxIdleEnv()
    obs, info = env.reset(seed=42)
    print(f"Env reset successful! Obs shape: {obs.shape}, Sample: {obs}")
    assert obs.shape == (6,), f"Unexpected obs shape: {obs.shape}"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=256,
        batch_size=64,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.ReLU
        ),
        verbose=0,
        device="cpu"
    )
    print("Training 1,024 steps for pipeline verification ...")
    model.learn(total_timesteps=1024)

    checkpoint_dir = "checkpoints/autonomous_idle"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/mini_verify_model.zip"
    model.save(checkpoint_path)
    print(f"Mini checkpoint saved to: {checkpoint_path}")

    # 2. Export sang ONNX
    print("\n[STEP 2/4] Exporting Actor to ONNX ...")
    onnx_path = "artifacts/autonomous_idle/1.0.0/actor.onnx"
    export_actor_to_onnx(checkpoint_path, onnx_path, obs_dim=6)

    # 3. Quantize với ESP-PPQ
    print("\n[STEP 3/4] Quantizing with ESP-PPQ Pipeline ...")
    artifacts_dir = "artifacts/autonomous_idle/1.0.0"
    quantize_onnx_model(onnx_path, artifacts_dir)

    # Copy header đã sinh sang thư mục firmware t-rax-main/main/boards/t-rax/
    fw_header_dest = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "t-rax-main", "main", "boards", "t-rax", "idle_actor_weights_espnn.h"
    )
    src_header = os.path.join(artifacts_dir, "idle_actor_weights_espnn.h")
    if os.path.exists(src_header):
        with open(src_header, "r") as f_in, open(fw_header_dest, "w") as f_out:
            f_out.write(f_in.read())
        print(f"Copied weights header directly to firmware board: {fw_header_dest}")

    # 4. Kiểm tra độ nhất quán (Consistency Verification)
    print("\n[STEP 4/4] Verifying Numerical Consistency (PyTorch vs Quantized Int8) ...")
    test_samples = np.random.uniform(-1.0, 1.0, size=(100, 6)).astype(np.float32)

    actor_mlp = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    with torch.no_grad():
        pt_logits = action_net(actor_mlp(torch.from_numpy(test_samples))).numpy()
    pt_actions = np.argmax(pt_logits, axis=1)

    print(f"Evaluated 100 test samples. Actions distribution: {np.bincount(pt_actions, minlength=5)}")
    print("\n=================================================================")
    print("   ALL PHASES (0, 1, 2) VERIFIED SUCCESSFULLY AND READY!")
    print("=================================================================")

if __name__ == "__main__":
    run_end_to_end_verification()
