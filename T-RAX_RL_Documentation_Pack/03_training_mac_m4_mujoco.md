# Hướng dẫn train T-RAX RL trên Mac mini M4 + MuJoCo

## 1. Stack

```text
macOS / Apple Silicon
        │
        ├── Python
        ├── MuJoCo
        ├── Gymnasium
        ├── Stable-Baselines3
        └── PyTorch
```

MuJoCo chịu trách nhiệm physics.

Gymnasium cung cấp chuẩn Environment API.

Stable-Baselines3 cung cấp PPO baseline.

PyTorch cung cấp neural network backend.

> Với policy nhỏ, tốc độ physics và environment stepping thường quan trọng hơn việc ép toàn bộ pipeline chạy GPU.

---

## 2. Cài môi trường

Khuyến nghị Python 3.11 hoặc version tương thích với stack hiện tại.

```bash
brew install python@3.11

mkdir trax-rl
cd trax-rl

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install mujoco gymnasium stable-baselines3 torch numpy tensorboard
```

Test MuJoCo:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Test PyTorch MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

MuJoCo Python package chính thức có prebuilt binary và có thể cài trực tiếp từ PyPI.

---

## 3. Project layout

```text
trax-rl/
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   └── policies/
│       ├── idle_v1.yaml
│       ├── follow_v1.yaml
│       └── play_ball_v1.yaml
├── assets/
│   ├── trax/
│   │   ├── base.xml
│   │   └── scene.xml
│   └── objects/
├── trax_rl/
│   ├── envs/
│   ├── observations/
│   ├── rewards/
│   ├── actions/
│   ├── randomization/
│   ├── policies/
│   ├── evaluation/
│   └── export/
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── visualize.py
│   └── export.py
├── runs/
├── checkpoints/
└── artifacts/
```

---

## 4. MuJoCo model V1

Không cần CAD hoàn chỉnh.

Ví dụ:

```xml
<mujoco model="trax">
    <option timestep="0.01"/>

    <worldbody>
        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
            friction="1.0 0.01 0.001"/>

        <body name="robot" pos="0 0 0.12">
            <freejoint/>
            <geom
                name="body"
                type="box"
                size="0.12 0.10 0.05"
                mass="0.8"/>
        </body>
    </worldbody>
</mujoco>
```

Giai đoạn locomotion mới thêm wheel/track approximation.

Không mô phỏng từng mắt xích ở V1.

---

## 5. Gymnasium Environment

```python
import gymnasium as gym
import mujoco
import numpy as np


class TraxIdleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            "assets/trax/scene.xml"
        )
        self.data = mujoco.MjData(self.model)

        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomize_episode()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self._apply_action(action)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._reward(action)

        terminated = False
        truncated = self._episode_time_exceeded()

        info = self._get_info()

        return obs, reward, terminated, truncated, info
```

Gymnasium custom environment nên được kiểm tra bằng:

```python
from gymnasium.utils.env_checker import check_env

check_env(TraxIdleEnv())
```

---

## 6. PoC Autonomous Idle

Observation V1:

```text
0 person_visible
1 person_x
2 motion_level
3 head_pan
4 time_since_event
5 previous_action
```

Action:

```text
0 REST
1 LOOK_LEFT
2 LOOK_RIGHT
3 LOOK_CENTER
4 TRACK_PERSON
```

Reward:

```python
def compute_reward(state, action):
    reward = 0.0

    if state.person_visible:
        reward += 0.5 * (1.0 - abs(state.person_x))

        if action == TRACK_PERSON:
            reward += 0.5

    else:
        if action == REST:
            reward += 0.03

    if action == state.previous_action:
        reward -= 0.02

    reward -= 0.005 * state.energy_cost

    return reward
```

Reward trên chỉ là baseline, phải tune bằng evaluation.

---

## 7. PPO training

```python
from stable_baselines3 import PPO

from trax_rl.envs.idle import TraxIdleEnv


env = TraxIdleEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="runs/idle_v1/",
)

model.learn(
    total_timesteps=1_000_000,
)

model.save(
    "checkpoints/idle_v1/final"
)
```

Chạy:

```bash
python scripts/train.py --policy idle_v1
```

---

## 8. Vectorized environments

Khi environment ổn định, dùng nhiều env song song để tăng throughput.

```python
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(
    TraxIdleEnv,
    n_envs=8,
)
```

Không mặc định rằng càng nhiều env càng nhanh. Benchmark:

```text
1
2
4
8
16
```

và đo:

```text
environment steps/s
PPO updates/s
wall-clock training time
CPU usage
memory usage
```

Mac mini M4 nên được tối ưu theo benchmark thực tế thay vì chọn số worker theo cảm tính.

---

## 9. MPS

PyTorch hỗ trợ backend `mps` trên macOS Apple Silicon.

Test:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

Tuy nhiên với PPO + MLP nhỏ:

- environment simulation thường chiếm phần đáng kể;
- transfer/overhead có thể làm GPU không nhanh hơn CPU;
- hãy benchmark CPU và MPS.

Không xem MPS là điều kiện bắt buộc cho RL của T-RAX.

---

## 10. Logging

Mỗi run phải lưu:

```text
config
git commit
seed
reward curve
episode length
success rate
collision count
action distribution
policy checkpoint
evaluation video
```

Không chỉ xem `ep_rew_mean`.

Metric Idle nên có:

```text
attention_success_rate
unnecessary_motion_rate
repetition_rate
rest_ratio
person_tracking_ratio
collision_rate
energy_proxy
```

---

## 11. Evaluation

Dùng seed cố định:

```python
EVAL_SEEDS = [
    1001,
    1002,
    1003,
    1004,
    1005,
]
```

So sánh policy theo cùng bộ scenario.

Không chọn model chỉ vì reward training lớn hơn.

---

## 12. Curriculum

### Stage A — Head only

```text
REST
LOOK_LEFT
LOOK_RIGHT
LOOK_CENTER
TRACK_PERSON
```

### Stage B — Memory

Thêm:

```text
last_person_x
time_since_person_seen
```

### Stage C — Body turn

Thêm:

```text
TURN_LEFT
TURN_RIGHT
```

### Stage D — Movement

Thêm:

```text
MOVE_FORWARD_SHORT
MOVE_BACKWARD_SHORT
```

### Stage E — Obstacle

Thêm sensor + collision penalty.

### Stage F — Full randomization

Random noise, latency, motor/servo response.

---

## 13. Domain Randomization

Ví dụ:

```python
self.person_speed = self.np_random.uniform(0.2, 1.2)
self.sensor_noise = self.np_random.uniform(0.0, 0.05)
self.servo_gain = self.np_random.uniform(0.85, 1.15)
self.control_delay = self.np_random.uniform(0.02, 0.08)
```

Randomization phải có config và version.

---

## 14. Export Actor

Sau khi train:

```text
SB3 checkpoint
      ↓
Actor network
      ↓
ONNX / raw weights
      ↓
quantization
      ↓
C/C++ representation
      ↓
ESP-IDF
```

Critic không cần deploy.

Với network nhỏ:

```text
16 → 64 → 64 → 8
```

policy có thể rất nhỏ so với vision model.

---

## 15. Sim-to-Real rollout

Không bật full control ngay.

### Phase 1 — replay test

Feed observation thật đã log vào policy offline.

### Phase 2 — shadow mode

Policy chạy trên robot thật nhưng không điều khiển motor.

Log:

```text
observation
predicted action
current scripted action
safety state
```

### Phase 3 — head only

Cho policy điều khiển pan/tilt.

### Phase 4 — limited base movement

Giới hạn speed thấp.

### Phase 5 — full idle behavior

Vẫn giữ safety supervisor.

---

## 16. Command checklist

```bash
# Activate environment
source .venv/bin/activate

# Verify MuJoCo
python -c "import mujoco; print(mujoco.__version__)"

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# Train
python scripts/train.py --policy idle_v1

# Evaluate
python scripts/eval.py --policy idle_v1

# Export
python scripts/export.py --policy idle_v1
```
