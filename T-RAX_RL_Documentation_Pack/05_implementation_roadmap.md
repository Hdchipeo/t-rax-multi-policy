# Lộ trình triển khai T-RAX RL

## Giai đoạn 0 — Chuẩn hóa software architecture

### Mục tiêu

Tách firmware thành:

```text
Perception
WorldState
BehaviorManager
BehaviorExecutor
Safety
Drivers
```

### Hoàn thành khi

- IDLE scripted hiện tại chạy qua BehaviorManager.
- Motor/servo không bị policy gọi trực tiếp.
- WorldState có timestamp.
- Safety có priority cao nhất.

---

## Giai đoạn 1 — RL sandbox trên Mac mini M4

### Việc cần làm

- cài MuJoCo;
- cài Gymnasium;
- cài Stable-Baselines3;
- tạo BaseTraxEnv;
- test `check_env`;
- logging;
- deterministic seed.

### Output

```text
TRAXIdle-v0
```

---

## Giai đoạn 2 — Autonomous Idle Head-only

### Observation

```text
person_visible
person_x
motion_level
head_pan
time_since_event
previous_action
```

### Action

```text
REST
LOOK_LEFT
LOOK_RIGHT
LOOK_CENTER
TRACK_PERSON
```

### Mục tiêu

Policy học:

```text
không có sự kiện → nghỉ
người xuất hiện → nhìn
người di chuyển → track
người biến mất → phản ứng theo history
```

### Không có

- base movement;
- obstacle;
- navigation.

---

## Giai đoạn 3 — Idle + short-term memory

Thêm:

```text
last_person_x
time_since_person_seen
previous_2_actions
```

Đánh giá:

```text
repetition_rate
tracking reacquisition
unnecessary movement
```

---

## Giai đoạn 4 — Sim-to-Real head-only

### Shadow Mode

Policy chạy nhưng chỉ log.

### Limited Mode

Policy điều khiển servo với:

```text
angle limit
speed limit
timeout
```

### Release

So sánh:

```text
scripted idle
vs
RL idle
```

Metric:

```text
reaction consistency
repetition
human-perceived naturalness
servo activity
safety overrides
```

---

## Giai đoạn 5 — Locomotion primitive

MuJoCo:

```text
differential drive approximation
```

Thêm action:

```text
TURN_LEFT
TURN_RIGHT
MOVE_FORWARD_SHORT
MOVE_BACKWARD_SHORT
```

Trước khi train phải có:

- obstacle cutoff;
- speed limit;
- timeout;
- stop primitive.

---

## Giai đoạn 6 — PlayBallPolicy

Observation:

```text
ball_visible
ball_x
ball_size
estimated_distance
obstacle_distance
velocity
```

Reward:

```text
alignment
approach
ball displacement
collision penalty
energy penalty
```

---

## Giai đoạn 7 — FollowPersonPolicy

Observation:

```text
person_x
person_distance
visibility
obstacle_distance
velocity
```

Reward:

```text
center person
maintain target distance
smooth control
avoid collision
```

---

## Giai đoạn 8 — Shared policy pipeline

Chuyển project sang config-driven:

```text
train.py --config ...
eval.py --config ...
export.py --config ...
```

Hoàn thiện:

- observation registry;
- reward registry;
- action registry;
- randomization registry;
- model registry.

---

## Giai đoạn 9 — Search / Explore

SearchObject có thể composition:

```text
Search
 ├── Look
 ├── Turn
 ├── Explore
 └── Approach
```

Không cần train một mega-policy ngay.

---

## Giai đoạn 10 — Hierarchical behavior

Sau khi ít nhất 3–5 skill ổn định:

```text
High-Level Controller
       ↓
select skill
       ↓
Skill Policy
```

High-level controller có thể là:

- rule/FSM;
- Behavior Tree;
- LLM intent;
- RL selector.

Thứ tự nên thử:

```text
Rule/BT
→ LLM intent
→ Hierarchical RL
```

---

# Checklist cho mỗi policy mới

## Problem

- [ ] skill được định nghĩa rõ;
- [ ] success criterion rõ;
- [ ] failure criterion rõ.

## Observation

- [ ] chỉ dùng feature cần thiết;
- [ ] range xác định;
- [ ] normalization;
- [ ] version schema.

## Action

- [ ] action contract rõ;
- [ ] safety boundary;
- [ ] fallback action.

## Reward

- [ ] reward component tách riêng;
- [ ] log từng component;
- [ ] kiểm tra reward hacking.

## Training

- [ ] seed;
- [ ] checkpoint;
- [ ] tensorboard;
- [ ] curriculum;
- [ ] randomization.

## Evaluation

- [ ] deterministic scenario;
- [ ] noise scenario;
- [ ] edge case;
- [ ] safety case;
- [ ] baseline comparison.

## Deployment

- [ ] actor-only;
- [ ] manifest;
- [ ] quantization test;
- [ ] latency benchmark;
- [ ] shadow mode;
- [ ] rollback.

---

# KPI nền tảng

Không dùng duy nhất reward.

## System

```text
steps/s
training wall time
model size
inference latency
RAM
Flash
```

## Behavior

```text
success_rate
collision_rate
action repetition
energy proxy
safety override rate
task completion
```

## Sim-to-Real

```text
simulation success
shadow disagreement
real success
override rate
distribution drift
```

---

# Đích kiến trúc dài hạn

```text
                         ┌───────────────┐
                         │ LLM / Planner │
                         └───────┬───────┘
                                 │ intent
                                 ▼
                         Behavior Manager
                                 │
             ┌───────────────────┼──────────────────┐
             ▼                   ▼                  ▼
           Idle               Follow            PlayBall
             │                   │                  │
             ├──── Search / Explore / Dock ────────┤
             │                                      │
             └──────────── Skill Library ───────────┘
                                 │
                                 ▼
                         Behavior Executor
                                 │
                                 ▼
                         Safety Supervisor
                                 │
                                 ▼
                         ESP-IDF Hardware
```

Triết lý:

> AI chọn hành vi. Firmware giữ an toàn. Driver điều khiển phần cứng.
