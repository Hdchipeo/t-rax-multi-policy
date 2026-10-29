"""Evaluate the goal controller, learned residual, or a legacy 39D checkpoint."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from stable_baselines3 import PPO
from trax_rl.evaluation import evaluate_scenarios, environment_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policies/purposeful_v2.yaml")
    parser.add_argument("--model", help="Omit for the nominal goal-skill baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--output", default="reports/purposeful_v2/evaluation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model = PPO.load(args.model, device="cpu") if args.model else None
    options = environment_options(config)
    options["enable_domain_randomization"] = args.randomize
    result = evaluate_scenarios(model, env_options=options, seeds=args.seeds,
        steps=args.steps, output=args.output)
    for scenario in [f"S{i}" for i in range(8)]:
        episodes = [r for r in result["episodes"] if r["scenario"] == scenario]
        print(scenario, f"{sum(r['passed'] for r in episodes)}/{len(episodes)}",
              "reward", round(float(sum(r['reward'] for r in episodes)/len(episodes)),2),
              sorted(set(reason for r in episodes for reason in r["reasons"])))
    print("Acceptance:", "PASSED" if result["passed"] else "NEEDS TRAINING", "Report:", args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
