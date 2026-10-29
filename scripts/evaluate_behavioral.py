#!/usr/bin/env python3
"""Legacy 39D evaluation, using per-episode scenario gates."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from trax_rl.evaluation import evaluate_scenarios


def evaluate_behavioral_policy(model_path=None):
    if model_path and not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    model = PPO.load(model_path, device="cpu") if model_path else None
    result = evaluate_scenarios(model, env_options={"enable_domain_randomization":False},
        output="reports/legacy_behavioral/evaluation.json")
    for episode in result["episodes"]:
        print(episode["scenario"], episode["seed"],
              "PASS" if episode["passed"] else "FAIL", episode["reasons"])
    print("Simulation acceptance:", "PASSED" if result["passed"] else "NEEDS TRAINING")
    return result["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Omit for stationary baseline")
    args = parser.parse_args()
    return 0 if evaluate_behavioral_policy(args.model) else 1


if __name__ == "__main__":
    raise SystemExit(main())
