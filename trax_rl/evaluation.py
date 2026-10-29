"""Scenario acceptance on individual episodes; no aggregate can hide a failure."""
import json
from pathlib import Path
import numpy as np
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv


def evaluate_scenarios(model=None, *, env_options=None, seeds=(1001, 1002, 1003),
                       steps=500, scenarios=None, output=None):
    scenarios = scenarios or [f"S{i}" for i in range(8)]
    records = []
    for scenario in scenarios:
        for seed in seeds:
            options = dict(env_options or {})
            options.update(scenario=scenario, max_episode_steps=steps, randomize_scenarios=False)
            env = TraxBehavioralEnv(**options)
            reward_sum = 0.0
            transitions = []
            last_state = None
            try:
                obs, _ = env.reset(seed=int(seed))
                for step in range(steps):
                    action = (model.predict(obs, deterministic=True)[0] if model is not None
                              else np.zeros(5, dtype=np.float32))
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += reward
                    goal = info["goal_metrics"]
                    state = (goal["type"], goal["target_id"], goal["phase"])
                    if state != last_state:
                        transitions.append(dict(step=step, **goal))
                        last_state = state
                    if terminated or truncated:
                        break
                metrics = info["scenario_metrics"]
                passed, reasons = env.scenario_manager.evaluate_acceptance(metrics)
                records.append(dict(scenario=scenario, seed=int(seed), steps=step+1,
                    reward=reward_sum, passed=passed, reasons=reasons, metrics=metrics,
                    goals=info["goal_metrics"], safety=info["safety_metrics"],
                    transitions=transitions))
            finally:
                env.close()
    result = dict(controller_mode=(env_options or {}).get("controller_mode", "rl"),
                  seeds=list(seeds), steps_per_episode=steps,
                  domain_randomization=(env_options or {}).get("enable_domain_randomization", False),
                  pass_rate=float(np.mean([r["passed"] for r in records])),
                  mean_reward=float(np.mean([r["reward"] for r in records])),
                  passed=all(r["passed"] for r in records), episodes=records)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(path)
    return result


def environment_options(config):
    env = config.get("environment", {})
    return dict(xml_path=env.get("scene_xml"), frameskip=env.get("frameskip", 10),
        timestep=env.get("timestep"),
        max_episode_steps=env.get("max_episode_steps", 500),
        controller_mode=env.get("controller_mode", "rl"),
        action_mode=config.get("action", {}).get("type", "continuous"),
        enable_domain_randomization=env.get("domain_randomization", True),
        reward_components=config.get("reward", {}).get("components", {}),
        goal_options=config.get("goals", {}))
