# 02 — MuJoCo Environment

## Mục tiêu

Xây dựng simulator đủ chính xác để học behavior, không cố mô phỏng track dynamics hoàn hảo ở V1.

## Robot V1

```text
base
left/right track approximation
pan joint
tilt joint
camera frame
collision geometry
front-distance sensor abstraction
```

## Cấu trúc

```text
simulation/
├── assets/
│   ├── robot/
│   │   ├── trax_base.xml
│   │   ├── trax_head.xml
│   │   ├── actuators.xml
│   │   └── sensors.xml
│   ├── objects/
│   ├── humans/
│   └── furniture/
├── worlds/
│   ├── empty_room.xml
│   ├── object_room.xml
│   ├── dynamic_room.xml
│   ├── human_room.xml
│   └── mixed_room.xml
└── envs/
    └── trax_idle_env.py
```

## Curriculum

```text
Stage 0: Empty room
Stage 1: Static objects
Stage 2: Randomized objects
Stage 3: Moving object
Stage 4: Human abstraction
Stage 5: Mixed environment
```

## Frequency

```text
MuJoCo physics     250–500 Hz
low-level control   50–100 Hz
RL policy             5–10 Hz
```

Baseline:

```text
physics_dt = 0.002 s
policy_dt  = 0.100 s
```

## Episode

```text
V1 training:      30–60 s
advanced:        120–300 s
stress testing:    1–24 h equivalent
```

## Event scheduler

Randomize:
- event time;
- event type;
- location;
- intensity.

Ví dụ:

```python
events = [
    {"time": 10.0, "type": "sound"},
    {"time": 20.0, "type": "human_move"},
    {"time": 35.0, "type": "ball_roll"},
]
```

## Definition of Done

- spawn ổn định;
- pan/tilt đúng range;
- collision hoạt động;
- reset reproducible;
- random action không crash;
- chạy hàng nghìn episode tự động.
