#!/usr/bin/env python3
import os
import argparse
import json
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
import onnx
import onnxruntime as ort

class ActorOnly(nn.Module):
    """
    Trích xuất riêng mạng Actor từ Stable-Baselines3 PPO model:
    Input: [batch_size, obs_dim] (float32)
    Output: [batch_size, action_dim] (logits float32)
    """
    def __init__(self, policy_net: nn.Sequential, action_net: nn.Linear):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.policy_net(x)
        logits = self.action_net(features)
        return logits

def export_actor_to_onnx(model_path: str, output_onnx_path: str, obs_dim: int = None, is_continuous: bool = None):
    print(f"Loading SB3 checkpoint: {model_path} ...")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception:
        model = SAC.load(model_path, device="cpu")
    actual_dim = int(model.observation_space.shape[0])
    if obs_dim is not None and obs_dim != actual_dim:
        raise ValueError(f"Observation mismatch: checkpoint={actual_dim}, requested={obs_dim}")
    obs_dim = actual_dim
    actual_continuous = hasattr(model.action_space, "low")
    if is_continuous is not None and is_continuous != actual_continuous:
        raise ValueError("Requested action type does not match checkpoint")
    is_continuous = actual_continuous
    if isinstance(model, SAC):
        raise ValueError("SAC requires tanh/unscale export; this exporter currently supports PPO only")

    # Trích xuất policy_net và action_net từ PyTorch module
    if hasattr(model.policy, "mlp_extractor"):
        actor_mlp = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net
    elif hasattr(model.policy, "actor"):
        actor_mlp = model.policy.actor.latent_pi
        action_net = model.policy.actor.mu
    else:
        raise ValueError("Unsupported policy network structure for ONNX export")

    actor_model = ActorOnly(actor_mlp, action_net)
    actor_model.eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    Path(output_onnx_path).parent.mkdir(parents=True, exist_ok=True)

    output_name = "action_continuous" if is_continuous else "action_logits"
    print(f"Exporting Actor model to ONNX: {output_onnx_path} (obs_dim={obs_dim}, output={output_name}) ...")
    torch.onnx.export(
        actor_model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=[output_name],
        dynamic_axes={"observation": {0: "batch_size"}, output_name: {0: "batch_size"}}
    )

    # 1. Kiểm định cấu trúc graph ONNX
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verification PASSED!")

    # 2. Kiểm định suy luận bằng ONNX Runtime
    session = ort.InferenceSession(output_onnx_path)
    test_input = np.random.default_rng(1001).uniform(-1.0, 1.0, size=(32, obs_dim)).astype(np.float32)
    
    with torch.no_grad():
        pytorch_output = actor_model(torch.from_numpy(test_input)).numpy()
    
    ort_inputs = {session.get_inputs()[0].name: test_input}
    ort_output = session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pytorch_output - ort_output))
    print(f"PyTorch vs ONNXRuntime Max Absolute Error: {max_diff:.8f}")
    assert max_diff < 1e-5, f"ONNX numerical deviation too large: {max_diff}"
    print(f"Successfully exported and validated ONNX Actor model to: {output_onnx_path}")
    metadata_path = Path(model_path).with_suffix(".metadata.json")
    training = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    checksum = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if training and training.get("checkpoint_sha256") != checksum:
        raise ValueError("Checkpoint metadata checksum mismatch")
    metadata = dict(checkpoint_sha256=checksum, num_timesteps=model.num_timesteps,
        observation_dim=obs_dim, controller_mode=training.get("controller_mode", "rl"),
        output_semantics="unclipped_actor_mean" if is_continuous else "action_logits",
        action_low=model.action_space.low.tolist() if is_continuous else None,
        action_high=model.action_space.high.tolist() if is_continuous else None,
        onnx_max_error=float(max_diff), validation_passed=training.get("validation_passed", False),
        requires_goal_runtime=training.get("controller_mode") == "goal_residual",
        deployment_ready=False)
    Path(output_onnx_path).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Export SB3 Actor to ONNX")
    parser.add_argument("--model", type=str, default="checkpoints/autonomous_behavioral_dynamics/final_model.zip", help="Path to SB3 zip")
    parser.add_argument("--output", type=str, default="artifacts/autonomous_behavioral_dynamics/1.0.0/actor.onnx", help="Output onnx path")
    parser.add_argument("--obs-dim", type=int, default=None, help="Optional check; defaults to checkpoint dimension")
    parser.add_argument("--discrete", action="store_true", help="Set for discrete action space (logits)")
    args = parser.parse_args()

    export_actor_to_onnx(args.model, args.output, args.obs_dim, is_continuous=not args.discrete)

if __name__ == "__main__":
    main()
