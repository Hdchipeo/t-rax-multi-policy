#!/usr/bin/env python3
"""
T-RAX Autonomous Behavioral Dynamics: Training Pipeline (Doc 08 - Training Pipeline)
Huấn luyện mạng nơ-ron PPO / SAC với:
- Môi trường: TraxBehavioralEnv (Vật lý MuJoCo, 39-dim Observation, 5D Action)
- 5 Homeostatic Internal Drives (Curiosity, Boredom, Social Need, Energy, Confidence)
- Episodic Novelty & Anti-Stare Decay (Doc 05)
- HumanAgent & 5-tier Attention Dynamics (Doc 06)
- BehavioralTelemetryCallback: Giám sát phân rã 11 thành phần thưởng phạt trên TensorBoard
"""

import os
import sys
import argparse
import json
import hashlib
from pathlib import Path
import yaml
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv
from trax_rl.evaluation import evaluate_scenarios, environment_options


def runtime_checksum():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "trax_rl").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class BehavioralValidationCallback(BaseCallback):
    """Select checkpoint by acceptance first, reward second, on held-out seeds."""
    def __init__(self, config, output_dir, n_envs):
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir)
        self.eval_config = config.get("evaluation", {})
        self.frequency = max(1, self.eval_config.get("frequency", 25000) // n_envs)
        self.best_score = (-1.0, -float("inf"))
        best_metadata = self.output_dir / "best_model.metadata.json"
        best_validation = self.output_dir / "best_validation.json"
        if best_metadata.exists() and best_validation.exists():
            metadata = json.loads(best_metadata.read_text())
            if metadata.get("runtime_sha256") == runtime_checksum() and metadata.get("config") == config:
                result = json.loads(best_validation.read_text())
                self.best_score = (result["pass_rate"], result["mean_reward"])
        self.last_evaluation_step = -1

    def evaluate(self):
        options = environment_options(self.config)
        options["enable_domain_randomization"] = self.eval_config.get("domain_randomization", True)
        result = evaluate_scenarios(self.model, env_options=options,
            seeds=self.eval_config.get("seeds", [1001, 1002, 1003]),
            steps=self.eval_config.get("steps", 500),
            output=self.output_dir / "latest_validation.json")
        self.last_evaluation_step = self.num_timesteps
        score = (result["pass_rate"], result["mean_reward"])
        self.logger.record("validation/pass_rate", result["pass_rate"])
        self.logger.record("validation/mean_reward", result["mean_reward"])
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.output_dir / "best_model.zip"))
            (self.output_dir / "best_validation.json").write_text(json.dumps(result, indent=2))
            save_training_metadata(self.output_dir / "best_model.zip", self.config, self.model, result)
        print(f"[validation {self.num_timesteps:,}] acceptance={result['pass_rate']:.1%}, reward={result['mean_reward']:.2f}")
        return result

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.evaluate()
        return True


def save_training_metadata(path, config, model, validation=None):
    path = Path(path)
    payload = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        num_timesteps=model.num_timesteps, observation_dim=int(model.observation_space.shape[0]),
        controller_mode=config.get("environment", {}).get("controller_mode", "rl"),
        config=config, validation_passed=bool(validation and validation["passed"]))
    payload["runtime_sha256"] = runtime_checksum()
    payload["effective_training"] = dict(n_envs=model.n_envs,
        n_steps=getattr(model, "n_steps", None), seed=model.seed,
        learning_rate=model.learning_rate if isinstance(model.learning_rate, (int,float)) else "schedule")
    path.with_suffix(".metadata.json").write_text(json.dumps(payload, indent=2))

class BehavioralTelemetryCallback(BaseCallback):
    """
    Callback giám sát chuyên sâu cho Autonomous Behavioral Dynamics (Doc 04, Doc 08).
    Ghi nhận:
    - 11 thành phần thưởng / phạt riêng biệt
    - 5 Động lực nội tại (Homeostatic Drives)
    - Tỷ lệ can thiệp của Lớp an toàn (Safety Override Rate)
    - Chỉ số độ mới Episodic Novelty
    """
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                def average(group):
                    keys = set().union(*(i.get(group, {}).keys() for i in infos))
                    return {key:float(np.mean([i[group][key] for i in infos
                        if key in i.get(group, {}) and isinstance(i[group][key], (int, float))]))
                        for key in keys if any(isinstance(i.get(group, {}).get(key), (int,float)) for i in infos)}
                breakdown = average("reward_breakdown")
                drives = average("internal_drives")
                safety = average("safety_metrics")
                novelty = average("novelty_metrics")

                # Log reward components to TensorBoard
                for k, v in breakdown.items():
                    self.logger.record(k, v)

                # Log internal drives to TensorBoard
                for k, v in drives.items():
                    self.logger.record(f"drives/{k}", v)

                # Log safety and novelty
                self.logger.record("safety/override_rate", safety.get("safety_override_rate", 0.0))
                self.logger.record("safety/action_changed_rate", safety.get("action_changed_rate", 0.0))
                for k, v in average("goal_metrics").items():
                    self.logger.record(f"goals/{k}", v)
                self.logger.record("novelty/mean_novelty", novelty.get("mean_novelty", 0.0))

                if self.verbose > 0:
                    print(f"[{self.num_timesteps:7d} steps] "
                          f"R_Total: {breakdown.get('reward/total', 0.0):+5.2f} | "
                          f"Nov: {breakdown.get('reward/novelty', 0.0):+5.2f} | "
                          f"Eng: {breakdown.get('reward/engagement', 0.0):+5.2f} | "
                          f"Quiet: {breakdown.get('reward/quiet', 0.0):+5.2f} | "
                          f"Jerk_Pen: {breakdown.get('penalty/jerk', 0.0):+5.2f} | "
                          f"Curiosity: {drives.get('curiosity', 0.0):.2f}")
        return True

import time

class LiveRenderCallback(BaseCallback):
    """
    Callback mở MuJoCo 3D Viewer và đồng bộ hình ảnh trực tiếp trong lúc huấn luyện (Live Debugging).
    """
    def __init__(self, render_fps: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.render_interval = 1.0 / max(1, render_fps)
        self.last_render_time = 0.0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        if hasattr(env, "render"):
            env.render()
        
        now = time.time()
        elapsed = now - self.last_render_time
        if elapsed < self.render_interval:
            time.sleep(self.render_interval - elapsed)
        self.last_render_time = time.time()
        return True

def make_env_thunk(rank: int, seed: int, action_mode: str = "continuous", scenario: str = "S7", randomize_scenarios: bool = True, env_options=None):
    def _init():
        options = dict(env_options or {})
        options.update(action_mode=action_mode, scenario=scenario, randomize_scenarios=randomize_scenarios)
        env = Monitor(TraxBehavioralEnv(**options))
        env.reset(seed=seed + rank * 100)
        return env
    return _init

def train_behavioral_policy(config_path: str = "configs/policies/purposeful_v2.yaml",
                            override_timesteps: int = None,
                            override_lr: float = None,
                            override_n_envs: int = None,
                            override_vec_type: str = None,
                            pretrained_model: str = None,
                            scenario: str = "S7",
                            randomize_scenarios: bool = True,
                            seed: int = 42,
                            verbose: int = 1,
                            gui: bool = False,
                            render_fps: int = 20):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_name = cfg["policy"].get("name", "autonomous_behavioral_dynamics")
    total_timesteps = override_timesteps if override_timesteps is not None else cfg["training"].get("total_timesteps", 1_000_000)
    if total_timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if pretrained_model and not Path(pretrained_model).is_file():
        raise FileNotFoundError(pretrained_model)
    action_type = cfg["action"].get("type", "continuous")
    n_envs = 1 if gui else (override_n_envs or cfg["training"].get("n_envs", 4))
    chosen_vec_type = "dummy" if (gui or n_envs == 1) else (override_vec_type or cfg["training"].get("vec_type", "subproc"))
    default_lr = 3e-5 if pretrained_model else cfg["training"].get("learning_rate", 3e-4)
    lr = override_lr or default_lr
    net_arch = cfg["network"].get("net_arch", [128, 128])
    algorithm = cfg["training"].get("algorithm", "PPO").upper()

    print("==================================================================")
    print("🚀 T-RAX AUTONOMOUS BEHAVIORAL DYNAMICS TRAINING PIPELINE")
    print(f"📦 Policy Name : {policy_name}")
    print(f"🧠 Algorithm   : {algorithm} | Action Space: {action_type} (5D)")
    print(f"👁 Observation : {cfg['observation']['dim']}-dim Vector | Net Arch: {net_arch}")
    print(f"🌍 Scenario    : {'RANDOMIZED S0-S7' if randomize_scenarios else scenario}")
    print(f"⚡ Timesteps   : {total_timesteps:,} | Parallel Envs: {n_envs} ({chosen_vec_type.upper()}) | Seed: {seed} | LR: {lr}")
    if gui:
        print(f"🖥️ MuJoCo 3D UI : ENABLED (Target {render_fps} FPS) | Mode: Single-Env Live Debugging")
        print("💡 Hint: Trên macOS, để tương tác mượt mà qua Cocoa UI, chạy lệnh: ./.venv/bin/mjpython scripts/train_behavioral.py --gui")
    if pretrained_model:
        print(f"🔄 Continuing from Checkpoint: {pretrained_model}")
    print("==================================================================")

    os.makedirs(f"runs/{policy_name}", exist_ok=True)
    os.makedirs(f"checkpoints/{policy_name}", exist_ok=True)

    # Khởi tạo parallel vectorized environments
    env_options = environment_options(cfg)
    env_fns = [make_env_thunk(i, seed, action_mode=action_type, scenario=scenario,
        randomize_scenarios=randomize_scenarios, env_options=env_options) for i in range(n_envs)]
    if chosen_vec_type == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    if vec_env.observation_space.shape != (cfg["observation"]["dim"],):
        vec_env.close()
        raise ValueError("Configured observation schema does not match environment")
    expected_dt = cfg["environment"].get("timestep", .01) * cfg["environment"].get("frameskip", 10)
    if not np.isclose(expected_dt, 1.0 / cfg["policy"].get("control_hz", 10)):
        vec_env.close()
        raise ValueError("Environment timestep/frameskip must match control_hz")

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ReLU
    )

    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Loading weights from {pretrained_model} with fine-tuning LR={lr}...")
        loader = PPO if algorithm == "PPO" else SAC
        try:
            model = loader.load(pretrained_model, env=vec_env, device="cpu", custom_objects={"learning_rate": lr})
        except Exception:
            vec_env.close()
            raise
        # SB3 adds num_timesteps itself when reset_num_timesteps=False.
        actual_total_timesteps = total_timesteps
        print(f"Resuming training: current {model.num_timesteps:,} -> target {model.num_timesteps + total_timesteps:,} steps")
    elif algorithm == "PPO":
        actual_total_timesteps = total_timesteps
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr,
            n_steps=cfg["training"].get("n_steps", 2048),
            batch_size=cfg["training"].get("batch_size", 64),
            n_epochs=cfg["training"].get("n_epochs", 10),
            gamma=cfg["training"].get("gamma", 0.99),
            gae_lambda=cfg["training"].get("gae_lambda", 0.95),
            clip_range=cfg["training"].get("clip_range", 0.2),
            ent_coef=cfg["training"].get("ent_coef", 0.01),
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=f"runs/{policy_name}",
            device="cpu",
            seed=seed
        )
    elif algorithm == "SAC":
        actual_total_timesteps = total_timesteps
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr,
            buffer_size=cfg["training"].get("buffer_size", 200_000),
            batch_size=cfg["training"].get("batch_size", 256),
            gamma=cfg["training"].get("gamma", 0.99),
            tau=0.005,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
            verbose=0,
            tensorboard_log=f"runs/{policy_name}",
            device="cpu",
            seed=seed
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    telemetry_callback = BehavioralTelemetryCallback(log_freq=200, verbose=verbose)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1000),
        save_path=f"checkpoints/{policy_name}",
        name_prefix="behavioral_checkpoint"
    )

    callbacks = [telemetry_callback, checkpoint_callback]
    validation_callback = BehavioralValidationCallback(cfg, f"checkpoints/{policy_name}", n_envs)
    callbacks.append(validation_callback)
    if gui:
        live_render_cb = LiveRenderCallback(render_fps=render_fps, verbose=verbose)
        callbacks.append(live_render_cb)

    print(f"\nStarting training loop (additional: {actual_total_timesteps:,}, target: {model.num_timesteps + actual_total_timesteps:,} steps)...")
    try:
        model.learn(
            total_timesteps=actual_total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False
        )
        validation = validation_callback.evaluate()
    finally:
        vec_env.close()

    final_model_path = f"checkpoints/{policy_name}/final_model.zip"
    model.save(final_model_path)
    save_training_metadata(final_model_path, cfg, model, validation)
    print(f"\n✅ Training finished successfully! Final model saved to: {final_model_path}")
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train T-RAX Autonomous Behavioral Dynamics Policy")
    parser.add_argument("--config", type=str, default="configs/policies/purposeful_v2.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint (.zip)")
    parser.add_argument("--scenario", type=str, default="S7", help="Specific scenario or S7 default")
    parser.add_argument("--no-randomize", action="store_true", help="Disable random scenario sampling on reset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments (e.g. 4, 8)")
    parser.add_argument("--vec-type", choices=["subproc", "dummy"], default=None, help="Vectorized environment backend: 'subproc' or 'dummy'")
    parser.add_argument("--gui", "--render", action="store_true", help="Open real-time MuJoCo 3D viewer window during training for visual debugging")
    parser.add_argument("--render-fps", type=int, default=20, help="Rendering frame rate for live viewer")
    args = parser.parse_args()

    train_behavioral_policy(
        config_path=args.config,
        override_timesteps=args.timesteps,
        override_lr=args.lr,
        override_n_envs=args.n_envs,
        override_vec_type=args.vec_type,
        pretrained_model=args.pretrained,
        scenario=args.scenario,
        randomize_scenarios=not args.no_randomize,
        seed=args.seed,
        verbose=args.verbose,
        gui=args.gui,
        render_fps=args.render_fps
    )

if __name__ == "__main__":
    main()
