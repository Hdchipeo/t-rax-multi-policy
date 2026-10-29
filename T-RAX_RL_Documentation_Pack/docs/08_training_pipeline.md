# 08 — Training Pipeline

## Stack

```text
MuJoCo
Gymnasium
Stable-Baselines3
PyTorch
TensorBoard
```

## Baseline A — SAC

Phù hợp continuous action, off-policy, replay buffer, entropy objective.

## Baseline B — PPO

Dùng làm đối chứng, đặc biệt với parallel envs.

## SAC prototype

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([make_env])

model = SAC(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    verbose=1,
)

model.learn(total_timesteps=2_000_000)
```

## Curriculum

```text
0 Motor sanity
1 Visual exploration
2 Object curiosity
3 Dynamic object
4 Sound event
5 Human presence
6 Attention seeking
7 Mixed environment
```

## Experiment metadata

Lưu:

```text
algorithm
seed
git_commit
env_version
reward_version
observation_version
action_version
hyperparameters
checkpoint
metrics
```

## Naming

```text
idle_sac_obs-v1_reward-v2_seed-03
```

## Multiple seeds

Ít nhất 3–5 seeds cho benchmark quan trọng.
