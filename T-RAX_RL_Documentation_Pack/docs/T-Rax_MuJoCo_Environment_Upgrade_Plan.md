# T-Rax MuJoCo Environment Upgrade Plan

## 1. Mục tiêu

Tài liệu này tổng hợp hiện trạng môi trường MuJoCo của robot T-Rax và đề xuất lộ trình nâng cấp để phục vụ **Autonomous Behavior + Multi-Policy Reinforcement Learning**.

Mục tiêu chính:

- Biến môi trường hiện tại từ một sandbox đơn giản thành **behavior laboratory**.
- Tách rõ:
  - physics,
  - perception,
  - internal state,
  - behavior selection,
  - RL policy,
  - safety layer,
  - scenario,
  - evaluation.
- Giữ mỗi RL policy tập trung vào một hành vi cụ thể.
- Cho phép mở rộng dần từ idle exploration đến object interaction, fear, curiosity, habituation và social behavior.
- Chuẩn bị tốt hơn cho Sim-to-Real từ MuJoCo sang robot T-Rax thật.

---

# 2. Hiện trạng môi trường

## 2.1 Thành phần đang có

Môi trường hiện tại đã có nền tảng tốt cho giai đoạn prototype:

- arena có tường bao,
- tracked robot,
- pan/tilt head,
- ball,
- obstacle,
- object tĩnh,
- collision,
- clearance,
- novelty,
- internal drives,
- reward,
- safety intervention.

Log hiện tại ghi nhận các trường chính như:

```text
Base
Yaw
Head Pan/Tilt
Ball position
Clearance
Novelty
Drives
Tier
Reward
```

Ví dụ:

```text
Base: (+0.87, +0.37)
ψ:-638°
Head: [P:103° T:103°]
Ball: (-0.32,+0.19,0.08)
Clr: 1.33m
Nov: 0.13
Drives: [C:0.10 B:0.00 S:0.93]
Tier 0 (Quiet)
Rwd: +0.58
```

---

# 3. Các vấn đề cần ưu tiên xử lý

## 3.1 Internal drive chưa phản ánh động lực thực tế

Nếu `S` đại diện cho Safety/Fear drive thì log hiện tại cho thấy giá trị này có xu hướng tăng dần theo thời gian.

Không nên thiết kế:

```text
time
 ↓
safety/fear ↑
```

Nên dùng:

```text
threat detected
 ↓
safety/fear ↑

safe environment
 ↓
safety/fear decay
```

Ví dụ:

```python
fear += (threat_gain * threat
         - recovery_rate * fear) * dt
fear = clamp(fear, 0.0, 1.0)
```

Internal drive cần là **dynamic state**, không phải timer.

---

## 3.2 Yaw đang tích lũy vượt ±360°

Log có thể xuất hiện:

```text
-555°
-608°
-642°
```

hoặc:

```text
+310°
+326°
+352°
```

Physics không nhất thiết sai, nhưng không nên feed trực tiếp giá trị yaw raw vào policy.

Nên encode:

```python
obs_yaw = [
    math.sin(yaw),
    math.cos(yaw)
]
```

Lợi ích:

- loại bỏ discontinuity tại ±180°,
- giúp policy generalize tốt hơn,
- tránh numerical distance giả giữa hai góc gần nhau.

---

## 3.3 Reward đang thiếu khả năng giải thích

Hiện tại chỉ log:

```text
Rwd: +0.42
```

Không đủ để biết robot được thưởng vì lý do gì.

Nên phân rã reward:

```text
Reward:
  novelty        +0.20
  exploration    +0.12
  object_interest +0.08
  smooth_motion  +0.04
  near_obstacle  -0.06
  energy_cost    -0.03

Total            +0.35
```

Điều này rất quan trọng để phát hiện:

- reward hacking,
- policy học sai mục tiêu,
- penalty quá yếu,
- exploration reward quá lớn.

---

## 3.4 Log chưa trả lời câu hỏi “Tại sao robot làm vậy?”

Log hiện tại mô tả trạng thái nhưng chưa mô tả reasoning pipeline của controller.

Nên bổ sung:

```text
Behavior
Policy
Trigger
Raw Action
Safe Action
Safety Intervention
Reward Components
```

Ví dụ:

```text
Behavior:
  selected = EXPLORE
  reason   = curiosity > threshold

Policy:
  active = explore_v3

Action:
  raw  = [0.42, 0.31]
  safe = [0.30, 0.25]

Safety:
  intervention = SPEED_CLAMP
```

---

# 4. Kiến trúc mục tiêu

```text
Sensors / Simulation
        │
        ▼
┌─────────────────────┐
│ Perception          │
│                     │
│ object              │
│ ball                │
│ obstacle            │
│ novelty             │
│ human               │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Internal State      │
│                     │
│ curiosity           │
│ boredom             │
│ fear / safety       │
│ social              │
│ energy              │
│ arousal             │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Behavior Manager    │
└─────────┬───────────┘
          │
     Policy Selector
          │
   ┌──────┼─────────┐
   ▼      ▼         ▼
Explore  Look    Approach
Policy   Policy   Policy
   │      │         │
   └──────┴────┬────┘
               ▼
        Action Arbitration
               │
               ▼
          Safety Layer
               │
               ▼
             MuJoCo
```

Nguyên tắc:

> Behavior Manager chọn hành vi.  
> RL policy giải quyết cách thực hiện hành vi.  
> Safety Layer luôn có quyền override.

---

# 5. Nâng cấp môi trường theo curriculum

## Level 0 — Physics Validation

Mục tiêu:

- xác thực differential/tracked drive,
- actuator,
- pan/tilt,
- contact,
- friction,
- collision,
- motor limits.

Scene:

```text
Robot
Ball
1 obstacle
Walls
```

Không train personality ở giai đoạn này.

---

## Level 1 — Navigation

Thêm:

- box,
- pillar,
- corridor,
- corner,
- dead-end,
- narrow gap,
- obstacle cluster.

Policy:

- `explore_policy`
- `avoid_obstacle_policy`
- `approach_target_policy`

---

## Level 2 — Object Interaction

Thêm:

- ball,
- unknown object,
- charging dock,
- moving object,
- human proxy.

Policy:

- `object_tracking_policy`
- `approach_object_policy`
- `ball_interaction_policy`
- `look_at_human_policy`

---

## Level 3 — Living Environment

Tạo môi trường gần với robot thật hơn:

```text
living room
desk
chair
door
wall
ball
human
unknown object
moving object
sound source
```

Mục tiêu:

```text
Idle
 ↓
Explore
 ↓
Notice
 ↓
Observe
 ↓
Approach
 ↓
Interact
 ↓
Lose interest
 ↓
Explore again
```

---

# 6. Scenario System

Scenario không nên hard-code trực tiếp vào một file MJCF duy nhất.

Cấu trúc đề xuất:

```text
assets/
    robot/
    objects/

worlds/
    empty_arena.xml
    obstacle_arena.xml
    living_room.xml

scenarios/
    quiet_room.yaml
    novel_object.yaml
    ball_curiosity.yaml
    obstacle_escape.yaml
    threat_event.yaml
    human_interaction.yaml
```

Ví dụ:

```yaml
scenario: ball_curiosity

spawn:
  robot:
    position: random

  ball:
    position: random
    distance_from_robot: [0.8, 2.5]

events:
  - time: random
    type: move_ball

success:
  - robot_looks_at_ball
  - robot_approaches_ball

failure:
  - collision
  - stuck_timeout
```

Scenario Manager chịu trách nhiệm:

```text
spawn
reset
events
stimulus
goals
termination
randomization
```

---

# 7. Bộ scenario nên phát triển

## 7.1 Quiet Room

Không có event đặc biệt.

Robot có thể:

- đứng yên,
- nhìn quanh,
- quay đầu,
- xoay base nhẹ,
- di chuyển ngắn.

Mục tiêu:

> Idle behavior có ý nghĩa thay vì random motion.

Reward gợi ý:

```python
reward = (
    0.30 * novelty_gain
    + 0.15 * observation_gain
    - 0.20 * energy_cost
    - 0.30 * repetitive_behavior
)
```

---

## 7.2 Novel Object Appears

Flow:

```text
Idle
 ↓
New object appears
 ↓
Novelty ↑
 ↓
Curiosity ↑
 ↓
LOOK
 ↓
APPROACH
```

Behavior Manager quyết định switch.

RL policy chỉ giải quyết:

```text
How to look?
How to approach?
```

---

## 7.3 Ball Curiosity

Flow:

```text
Detect ball
 ↓
Orient head
 ↓
Rotate base
 ↓
Approach
 ↓
Inspect
 ↓
Push
 ↓
Follow
```

Nên tách:

```text
ball_track_policy
ball_approach_policy
ball_push_policy
```

---

## 7.4 Threat / Startle

Stimulus:

- vật thể lao tới,
- chuyển động bất ngờ,
- tiếng động lớn,
- collision bất ngờ.

Threat estimator:

```text
threat =
    proximity
  + relative_velocity
  + sudden_audio
  + unexpected_contact
```

Behavior:

```text
LOW
  → orient

MEDIUM
  → stop + observe

HIGH
  → back away

CRITICAL
  → escape
```

---

## 7.5 Repeated Stimulus / Habituation

Object xuất hiện nhiều lần.

Ví dụ:

```text
First exposure:
Novelty = 1.0

Later exposures:
Novelty = 0.2
```

Mục tiêu:

> Robot bớt phản ứng với thứ đã quen.

Đây là nền tảng của memory + habituation.

---

## 7.6 Moving Human

Human proxy có thể bắt đầu từ capsule/cylinder.

Scenario:

```text
human enters
 ↓
robot detects
 ↓
head tracks
 ↓
social interest ↑
```

Sau đó thêm:

```text
slow approach → curiosity/social
fast approach → caution/fear
```

---

## 7.7 Competing Motivations

Ví dụ:

```text
ball nearby

Curiosity = 0.8
Safety    = 0.2

→ approach
```

Sau đó có threat:

```text
Curiosity = 0.8
Safety    = 0.9

→ retreat
```

Khi threat biến mất:

```text
Safety ↓
→ robot có quay lại ball hay không?
```

Đây là bài toán:

```text
Behavior Arbitration
```

---

# 8. Observation Space

Không nên phụ thuộc quá mạnh vào world coordinates.

Không nên:

```text
ball_world_x
ball_world_y
robot_world_yaw
```

Nên:

```text
ball_relative_x
ball_relative_y
ball_distance
ball_bearing
```

Ví dụ:

```python
obs = [
    linear_velocity,
    angular_velocity,

    math.sin(yaw),
    math.cos(yaw),

    pan,
    tilt,

    ball_visible,
    ball_distance,
    math.sin(ball_bearing),
    math.cos(ball_bearing),

    front_clearance,
    left_clearance,
    right_clearance,

    curiosity,
    boredom,
    safety,

    previous_action_left,
    previous_action_right,
]
```

Ưu điểm:

- policy dễ generalize,
- dễ chuyển sang robot thật,
- giảm phụ thuộc absolute map position.

---

# 9. Sensor Simulation

Hiện tại `Clearance` đơn lẻ chưa đủ cho behavior phức tạp.

Nên nâng thành:

```text
front
front-left
front-right
left
right
rear
```

Ví dụ:

```text
      FL   F   FR
        \ | /
 L ------ R ------ R
        /   \
       RL   RR
```

Sau này có thể map gần với hardware thật:

```text
MuJoCo virtual sensor
        ↓
Observation API
        ↑
ESP32 ToF sensor
```

Policy dùng cùng interface.

---

# 10. Camera Simulation

Camera nên gắn vào head:

```text
base
 └─ pan
     └─ tilt
         └─ camera
```

Giai đoạn đầu không nên feed raw RGB trực tiếp vào RL.

Pipeline tốt hơn:

```text
MuJoCo Camera
      ↓
Vision Emulator
      ↓
ball_visible
ball_bbox_x
ball_bbox_y
ball_size
object_class
      ↓
RL Policy
```

Robot thật:

```text
OV2640
 ↓
YOLO / ESP-DL
 ↓
same observation API
```

Như vậy Sim-to-Real dễ hơn đáng kể.

---

# 11. Domain Randomization

Mỗi episode nên random:

```text
robot position
robot heading
ball position
obstacle layout
object size
surface friction
motor scale
sensor noise
camera noise
lighting
moving object velocity
```

Ví dụ:

```python
friction = rng.uniform(0.7, 1.2)
motor_scale = rng.uniform(0.85, 1.15)
tof_noise = rng.normal(0.0, 0.01)
```

Mục tiêu:

> Policy học hành vi robust thay vì nhớ một scene cố định.

---

# 12. Internal State nâng cao

Các state nên cân nhắc:

```text
curiosity
boredom
fear
social_interest
energy
arousal
fatigue
novelty
habituation
```

Ví dụ arousal/heart-like state:

```python
arousal += (
    threat_gain * threat
    + novelty_gain * novelty
    - recovery_rate * arousal
) * dt
```

Arousal có thể ảnh hưởng:

- tốc độ,
- reaction latency,
- exploration willingness,
- behavior threshold,
- head motion,
- attention.

Internal state phải có ảnh hưởng lên decision-making, không chỉ dùng để animation.

---

# 13. Safety Layer

Safety Layer nên nằm ngoài RL.

Pipeline:

```text
RL Action
   │
   ▼
Safety Layer
   │
   ├─ distance clamp
   ├─ collision prediction
   ├─ emergency stop
   ├─ backoff escape
   └─ motor limit
   │
   ▼
Actuator Command
```

Nguyên tắc:

> RL có thể thử nghiệm. Safety Layer phải đảm bảo giới hạn vật lý.

Các event hiện tại như:

```text
OBSTACLE_PROXIMITY_CLAMP
OBSTACLE_BACKOFF_ESCAPE
```

nên tiếp tục giữ và mở rộng.

---

# 14. Logging mới

Format đề xuất:

```text
[Episode 12 | Step 320]

State:
  pose        = (0.42, -0.16)
  yaw         = 71 deg
  velocity    = 0.23 m/s

Head:
  pan         = 103 deg
  tilt        = 98 deg

Perception:
  ball_visible   = true
  ball_distance  = 0.72 m
  ball_bearing   = -18 deg
  novelty        = 0.64

Drives:
  curiosity = 0.72
  boredom   = 0.13
  safety    = 0.21
  arousal   = 0.30

Behavior:
  selected = APPROACH_OBJECT
  trigger  = curiosity_threshold

Policy:
  active = approach_object_v2

Action:
  raw_left   = 0.42
  raw_right  = 0.36
  safe_left  = 0.36
  safe_right = 0.31

Safety:
  intervention = SPEED_CLAMP

Reward:
  approach_progress = +0.18
  novelty           = +0.12
  smoothness        = +0.03
  energy            = -0.02
  obstacle          = -0.04

Total:
  +0.27
```

---

# 15. Metrics

Không nên dùng Episode Reward làm metric duy nhất.

Nên thêm:

```text
collision_count
safety_intervention_count
distance_travelled
area_coverage
object_discovery_count
object_revisit_count
stuck_time
energy_cost
behavior_switch_count
behavior_duration
head_motion_variance
novelty_gain
interaction_success
policy_success_rate
```

Reward dùng để train.

Metrics dùng để đánh giá.

Hai khái niệm này nên tách biệt.

---

# 16. Cấu trúc project đề xuất

```text
trax_rl/
│
├── assets/
│   ├── robot/
│   │   └── trax.xml
│   └── objects/
│       ├── ball.xml
│       ├── box.xml
│       └── human.xml
│
├── worlds/
│   ├── empty_arena.xml
│   ├── obstacle_arena.xml
│   └── living_room.xml
│
├── scenarios/
│   ├── quiet_room.yaml
│   ├── novel_object.yaml
│   ├── ball_curiosity.yaml
│   ├── threat_event.yaml
│   └── human_interaction.yaml
│
├── env/
│   ├── trax_env.py
│   ├── observation.py
│   ├── reward.py
│   └── randomization.py
│
├── behavior/
│   ├── manager.py
│   ├── drives.py
│   ├── arbitration.py
│   └── memory.py
│
├── policies/
│   ├── explore/
│   ├── look/
│   ├── approach/
│   ├── avoid/
│   └── ball/
│
├── safety/
│   └── safety_layer.py
│
└── evaluation/
    ├── metrics.py
    ├── evaluator.py
    └── scenarios.py
```

---

# 17. Roadmap triển khai

## Phase 1 — Refactor nền tảng

Ưu tiên:

1. Refactor log.
2. Reward decomposition.
3. Yaw → sin/cos.
4. Relative observation.
5. Tách Safety Layer.
6. Chuẩn hóa action/observation API.

---

## Phase 2 — Scenario Framework

Xây:

```text
ScenarioManager
ScenarioConfig
Event
Stimulus
Spawn
Reset
Termination
SuccessCondition
```

Scenario đầu tiên:

```text
QuietRoom
NovelObject
BallCuriosity
```

---

## Phase 3 — Policy đầu tiên

Train:

```text
ExplorePolicy
```

Observation:

```text
clearance
velocity
novelty
curiosity
boredom
previous action
```

Action:

```text
left motor
right motor
```

Mục tiêu:

```text
explore safely
avoid repetitive loops
increase area coverage
avoid getting stuck
```

---

## Phase 4 — Perception Policy

Train:

```text
LookAroundPolicy
ObjectTrackingPolicy
```

Action:

```text
pan
tilt
```

Sau đó kết hợp base + head bằng Behavior Manager.

---

## Phase 5 — Interaction

Train:

```text
ApproachObjectPolicy
BallTrackPolicy
BallApproachPolicy
BallPushPolicy
```

Không gom tất cả thành một policy.

---

## Phase 6 — Internal Drives

Bổ sung:

```text
curiosity
boredom
fear
arousal
social
habituation
```

Trước tiên dùng deterministic dynamics.

Sau đó mới thử learning một phần của drive model nếu thực sự cần.

---

## Phase 7 — Behavioral Arbitration

Behavior Manager chọn policy dựa trên:

```text
environment context
internal drives
current task
sensor state
LLM command
safety condition
```

Ví dụ:

```text
if critical_threat:
    ESCAPE

elif human_detected and social_interest > threshold:
    SOCIAL_LOOK

elif object_novelty > threshold:
    OBSERVE_OBJECT

elif boredom > threshold:
    EXPLORE

else:
    IDLE
```

RL không cần xử lý toàn bộ logic này.

---

## Phase 8 — Sim-to-Real

Randomize:

```text
friction
motor response
sensor noise
latency
camera detection error
wheel slip
mass
center of gravity
```

Sau đó deploy policy lên runtime:

```text
Observation
 ↓
Policy inference
 ↓
Safety Layer
 ↓
ESP32 motor/servo controller
```

---

# 18. Thứ tự ưu tiên thực tế

## P0 — Làm ngay

- reward decomposition,
- behavior/policy logging,
- action before/after safety,
- yaw sin/cos,
- robot-relative observations,
- safety intervention metrics.

## P1 — Tiếp theo

- Scenario Manager,
- QuietRoom,
- NovelObject,
- BallCuriosity,
- ExplorePolicy.

## P2 — Sau khi baseline ổn

- LookAroundPolicy,
- ObjectTrackingPolicy,
- ApproachPolicy,
- sensor noise,
- domain randomization.

## P3 — Autonomous behavior nâng cao

- fear,
- arousal,
- habituation,
- social interest,
- competing drives,
- long-term memory.

---

# 19. Kiến trúc dài hạn

```text
                Environment
                     │
                     ▼
                 Sensors
                     │
                     ▼
                Perception
                     │
                     ▼
              Internal State
                     │
                     ▼
             Behavior Manager
                     │
                     ▼
              Policy Selector
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Explore         Look        Approach
     Policy         Policy       Policy
       │             │             │
       └─────────────┼─────────────┘
                     ▼
             Action Arbitration
                     │
                     ▼
                Safety Layer
                     │
                     ▼
                   Robot
                     │
                     ▼
              Environment
```

Scenario Manager đứng ngoài vòng controller:

```text
Scenario Manager
      │
      ├── world setup
      ├── spawn
      ├── event
      ├── stimulus
      ├── reset
      └── evaluation
```

---

# 20. Kết luận

Môi trường hiện tại phù hợp để làm **T-Rax RL Sandbox v0**, nhưng chưa nên tăng độ phức tạp bằng cách chỉ thêm nhiều object hoặc scene đẹp hơn.

Bước nâng cấp quan trọng nhất là biến sandbox thành:

> **Behavior Laboratory**

với:

```text
Scenario
Stimulus
Internal State
Behavior Manager
Multi-Policy RL
Safety Layer
Reward Decomposition
Evaluation Metrics
Domain Randomization
```

Thứ tự phát triển nên là:

```text
Physics
 ↓
Observation
 ↓
Logging
 ↓
Safety
 ↓
Scenario
 ↓
Explore Policy
 ↓
Look / Track
 ↓
Approach
 ↓
Interaction
 ↓
Internal Drives
 ↓
Behavior Arbitration
 ↓
Personality / Mood
 ↓
Sim-to-Real
```

Không nên cố train một policy “biết tất cả”.

Kiến trúc Multi-Policy giúp T-Rax:

- dễ train,
- dễ debug,
- dễ thay policy,
- dễ đánh giá,
- dễ mở rộng behavior,
- dễ deploy lên embedded hardware,
- dễ kết hợp với Behavior Manager và LLM.

---

# Keywords

- **Multi-Policy Reinforcement Learning**: Học tăng cường nhiều policy
- **Behavior Manager**: Bộ điều phối hành vi
- **Behavior Arbitration**: Phân xử lựa chọn hành vi
- **Scenario Manager**: Bộ quản lý kịch bản
- **Stimulus**: Kích thích
- **Internal Drive**: Động lực nội tại
- **Curiosity**: Tò mò
- **Boredom**: Chán
- **Fear Drive**: Động lực sợ/nguy cơ
- **Arousal**: Mức kích hoạt sinh lý/nội tại
- **Habituation**: Quen dần với kích thích
- **Reward Shaping**: Thiết kế reward
- **Reward Decomposition**: Phân rã reward
- **Robot-Centric Observation**: Observation theo hệ tọa độ robot
- **Curriculum Learning**: Huấn luyện tăng dần độ khó
- **Domain Randomization**: Ngẫu nhiên hóa môi trường
- **Safety Intervention**: Can thiệp an toàn
- **Sim-to-Real**: Chuyển từ mô phỏng sang robot thật
