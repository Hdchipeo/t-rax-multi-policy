# Pipeline tối ưu cho các policy tương lai

## 1. Vấn đề cần tránh

Nếu mỗi policy mới tạo:

```text
new_env.py
new_reward.py
new_train.py
new_export.py
```

bằng copy-paste, project sẽ nhanh chóng khó bảo trì.

Giải pháp là biến project thành một **Policy Training Platform**.

---

## 2. Config-driven pipeline

Một policy nên chủ yếu được khai báo qua config.

Ví dụ:

```yaml
policy:
  name: play_ball
  version: 1

environment:
  id: TRAXPlayBall-v1
  control_hz: 20

observation:
  schema: ball_obs_v1

action:
  schema: differential_drive_v1

reward:
  components:
    approach: 1.0
    alignment: 0.3
    ball_motion: 2.0
    collision: -5.0
    energy: -0.01

network:
  hidden: [64, 64]

training:
  algorithm: PPO
  total_steps: 2000000
  learning_rate: 0.0003

randomization:
  profile: mobile_robot_v1
```

Training command không đổi:

```bash
python scripts/train.py --config configs/policies/play_ball_v1.yaml
```

---

## 3. Shared Environment Core

```text
BaseTraxEnv
├── physics
├── reset
├── timing
├── robot state
├── randomization
├── logging
└── rendering
```

Policy env chỉ thêm task:

```text
IdleEnv
PlayBallEnv
FollowEnv
ExploreEnv
```

Ví dụ:

```python
class BaseTraxEnv(gym.Env):
    ...

class PlayBallEnv(BaseTraxEnv):
    def task_observation(self):
        ...

    def task_reward(self):
        ...
```

---

## 4. Observation Registry

```python
OBSERVATION_REGISTRY = {
    "idle_obs_v1": build_idle_obs_v1,
    "ball_obs_v1": build_ball_obs_v1,
    "follow_obs_v1": build_follow_obs_v1,
}
```

Mỗi schema phải:

- có version;
- có dimension cố định;
- mô tả range;
- có normalization;
- có unit test.

Ví dụ tài liệu schema:

```text
idle_obs_v1

index  name                     range
0      person_visible           [0, 1]
1      person_x                 [-1, 1]
2      motion_level             [0, 1]
3      head_pan                 [-1, 1]
4      time_since_event         [0, 1]
5      previous_action_norm     [-1, 1]
```

---

## 5. Reward Components

Không hard-code reward lớn trong từng env.

Tạo component:

```text
AttentionReward
ApproachReward
AlignmentReward
NoveltyReward
CollisionPenalty
EnergyPenalty
RepetitionPenalty
SmoothnessReward
CoverageReward
```

Ví dụ:

```python
reward = RewardMixer([
    AttentionReward(weight=0.5),
    EnergyPenalty(weight=0.01),
    RepetitionPenalty(weight=0.1),
])
```

Lợi ích:

- reuse;
- ablation test;
- log từng reward;
- tune dễ hơn.

---

## 6. Action Schema

```text
idle_action_v1
drive_continuous_v1
head_continuous_v1
navigation_primitive_v1
```

Policy output không được implicit.

Manifest phải nói chính xác output có nghĩa gì.

---

## 7. Randomization Profiles

Ví dụ:

```text
head_robot_v1
mobile_robot_v1
vision_noise_v1
full_sim2real_v1
```

Một profile:

```yaml
motor_gain: [0.85, 1.15]
sensor_noise: [0.00, 0.05]
control_delay: [0.02, 0.08]
floor_friction: [0.7, 1.2]
```

Policy mới reuse profile cũ.

---

## 8. Unified Training CLI

Mục tiêu:

```bash
python scripts/train.py \
  --config configs/policies/follow_v2.yaml \
  --seed 42
```

Không tạo `train_follow.py`, `train_ball.py`, `train_idle.py` riêng.

---

## 9. Unified Evaluation

Evaluation suite:

```text
Scenario
├── nominal
├── noise
├── latency
├── edge_case
└── safety
```

Output:

```json
{
  "success_rate": 0.91,
  "collision_rate": 0.01,
  "mean_reward": 43.2,
  "energy_score": 0.72,
  "repetition_rate": 0.03
}
```

Policy phải pass threshold trước khi export.

---

## 10. Model Registry

Artifact layout:

```text
artifacts/
└── autonomous_idle/
    └── 1.2.0/
        ├── policy.zip
        ├── actor.onnx
        ├── actor_int8.bin
        ├── manifest.json
        ├── config.yaml
        ├── metrics.json
        └── eval_report.md
```

Không overwrite model cũ.

---

## 11. Promotion stages

```text
EXPERIMENT
   ↓
CANDIDATE
   ↓
SIM_VALIDATED
   ↓
SHADOW_VALIDATED
   ↓
HARDWARE_LIMITED
   ↓
RELEASE
```

Mỗi stage có gate.

Ví dụ `SIM_VALIDATED`:

```text
success_rate >= 0.90
collision_rate <= 0.01
no NaN output
passes deterministic scenarios
```

---

## 12. Regression testing

Mỗi thay đổi environment/reward phải chạy lại:

```text
policy baseline
vs
candidate
```

Không chỉ train model mới.

Phải biết:

```text
policy mới tốt hơn ở đâu
policy mới tệ hơn ở đâu
```

---

## 13. Seed policy

Training:

```text
5 seeds tối thiểu cho benchmark nghiêm túc
```

Development nhanh:

```text
1 seed
```

Release candidate:

```text
multi-seed evaluation
```

Không kết luận dựa trên một run may mắn.

---

## 14. Dataset bridge từ robot thật

Dù là RL, log robot thật vẫn rất có giá trị.

Schema:

```text
timestamp
world_state
policy_id
policy_action
safety_override
actual_motor_command
event
```

Dữ liệu này dùng để:

- replay;
- calibrate simulation;
- tìm distribution mismatch;
- tune domain randomization;
- imitation learning sau này.

---

## 15. Sim Calibration Loop

```text
Real Robot Logs
      ↓
parameter estimation
      ↓
MuJoCo calibration
      ↓
Domain Randomization update
      ↓
Retrain
      ↓
Hardware evaluation
      ↺
```

Đây là pipeline dài hạn quan trọng hơn việc chỉ tăng số timestep training.

---

## 16. Chuẩn hóa policy API

Một policy runtime chỉ cần:

```cpp
struct PolicyInput {
    const float* data;
    size_t size;
};

struct PolicyOutput {
    const float* data;
    size_t size;
};

class PolicyRuntime {
public:
    bool load(const PolicyManifest& manifest);
    bool infer(const PolicyInput& input, PolicyOutput& output);
};
```

Behavior layer chịu trách nhiệm diễn giải output.

---

## 17. Policy composition trong tương lai

Không merge network.

Cho phép Behavior Manager composition:

```text
SearchObject
   │
   ├── ExplorePolicy
   ├── LookPolicy
   └── ApproachPolicy
```

Task phức tạp được tạo bằng nhiều skill nhỏ.

Đây là bước trung gian rất tốt trước Hierarchical RL.
