# 12 — Implementation Roadmap

## Milestone 1 — MuJoCo V1

Implement:
- base;
- differential drive approximation;
- pan/tilt;
- collision;
- actuator limits;
- camera frame.

Done khi random action chạy ổn.

## Milestone 2 — Gymnasium IdleEnv

Implement:
- reset;
- step;
- observation V1;
- action V1;
- telemetry;
- event scheduler.

Target:

```text
check_env pass
10k random steps không crash
```

## Milestone 3 — IdlePolicy v0.1

Reward:
```text
smooth
view exploration
quiet
collision penalty
repeat penalty
energy penalty
```

Target behavior:
```text
scan
reorient
light movement
remain still
```

## Milestone 4 — Curiosity v0.2

Thêm:
```text
episodic memory
novelty score
novelty decay
object novelty
```

Chưa cần RND/ICM.

## Milestone 5 — Social v0.3

Thêm:
```text
HumanAgent
gaze
attention intensity
signal cost
cooldown
social-interest output
```

## Milestone 6 — Mixed Environment

Random:
```text
human
object
motion
sound
obstacles
```

## Milestone 7 — Evaluation

```text
multi-seed
scenario matrix
reward ablation
long-horizon tests
```

## Milestone 8 — Domain Randomization

Đo robot thật:
```text
servo latency
motor asymmetry
ToF noise
vision latency
```

## Milestone 9 — Export

```text
PyTorch
 ↓
ONNX
 ↓
quantization
 ↓
embedded runtime
```

Benchmark cả ESP32-S3 và ESP32-P4.

## Milestone 10 — Shadow Mode

Không actuator.

## Milestone 11 — Limited Real Execution

```text
head
 ↓
rotation
 ↓
low-speed base
 ↓
full bounded idle
```

## Milestone 12 — Multi-Policy Integration

```text
Idle
Tracking
Ball
Social
Approach
Avoid
```

## Sprint đầu tiên đề xuất

Tạo:

```text
simulation/assets/robot/trax_base.xml
simulation/worlds/empty_room.xml
simulation/envs/trax_idle_env.py
```

Action trước mắt:

```text
linear
angular
pan
tilt
```

Observation tối thiểu:

```text
pan
tilt
linear_velocity
angular_velocity
front_distance
```

Mục tiêu:

> chạy được T-Rax simulated qua Gymnasium bằng random action.

Sau đó mới thêm reward và train SAC.
