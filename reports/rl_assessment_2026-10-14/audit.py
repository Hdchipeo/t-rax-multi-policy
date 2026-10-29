"""Read-only policy evaluation; run from the repository root with .venv/bin/python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import random
import numpy as np
from stable_baselines3 import PPO
from trax_rl.envs.trax_behavioral_env import TraxBehavioralEnv

OUT = Path(__file__).parent
SEEDS = [42, 101, 2024]
STEPS = 500
results = {}
for name, path in {
    'baseline': None,
    'pre_causal_fix': 'checkpoints/autonomous_behavioral_dynamics/final_model_pre_causal_fix.zip',
    'final': 'checkpoints/autonomous_behavioral_dynamics/final_model.zip',
}.items():
    model = PPO.load(path, device='cpu') if path else None
    results[name] = {}
    for sc in [f'S{i}' for i in range(8)]:
        episodes = []
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            env = TraxBehavioralEnv(scenario=sc, max_episode_steps=STEPS,
                                    enable_domain_randomization=False)
            obs, info = env.reset(seed=seed)
            reward_sum = 0.
            quiet = collisions = changed = nonfinite = switches = 0
            distance = 0.
            previous_pos = env.data.qpos[:2].copy()
            last_behavior = None
            breakdown_sum = {}
            for step in range(STEPS):
                action = model.predict(obs, deterministic=True)[0] if model else np.array([0,0,0,0,.1], dtype=np.float32)
                obs, reward, term, trunc, info = env.step(action)
                reward_sum += reward
                safe = np.array(info['safety_metrics']['safe_action'])
                changed += int(np.max(np.abs(action - safe)) > 1e-5)
                quiet += int(np.sum(np.abs(safe[:4])) < .08 and safe[4] < .2)
                collisions += int(info['reward_breakdown']['penalty/collision'] > 0)
                nonfinite += int(not (np.all(np.isfinite(obs)) and np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)) and np.isfinite(reward)))
                behavior = info['safety_metrics']['current_behavior']
                switches += int(last_behavior is not None and behavior != last_behavior)
                last_behavior = behavior
                distance += float(np.linalg.norm(env.data.qpos[:2] - previous_pos))
                previous_pos = env.data.qpos[:2].copy()
                for key, val in info['reward_breakdown'].items():
                    breakdown_sum[key] = breakdown_sum.get(key, 0.) + val
                if term or trunc:
                    break
            metrics = info['scenario_metrics'].copy()
            passed, reasons = env.scenario_manager.evaluate_acceptance(metrics)
            metrics.update(seed=seed, reward=reward_sum, steps=step+1,
                           quiet_ratio_strict=quiet/(step+1), collision_steps=collisions,
                           action_changed_rate=changed/(step+1), nonfinite_steps=nonfinite,
                           behavior_switches=switches, distance_m=distance,
                           mujoco_warning_count=int(sum(w.number for w in env.data.warning)),
                           passed=passed, reasons=reasons,
                           reward_breakdown_mean={k:v/(step+1) for k,v in breakdown_sum.items()})
            episodes.append(metrics)
            env.close()
        means = {k:float(np.mean([ep[k] for ep in episodes])) for k,v in episodes[0].items()
                 if isinstance(v, (int,float)) and not isinstance(v, bool)}
        results[name][sc] = {'mean':means, 'episodes':episodes}
        print(name, sc, 'reward', round(means['reward'],2), 'jerk',round(means['motion_jerk'],4),
              'quiet',round(means['quiet_ratio_strict'],3), 'changed',round(means['action_changed_rate'],3),
              'passes',sum(ep['passed'] for ep in episodes),flush=True)
(OUT/'metrics.json').write_text(json.dumps({'seeds':SEEDS,'steps_per_episode':STEPS,
    'domain_randomization':False,'deterministic_policy':True,'global_rng_seeded':True,
    'results':results}, indent=2))
