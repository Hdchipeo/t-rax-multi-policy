# 13 — Behavioral Scenarios Implementation Plan

## 1\. Mục tiêu

Tài liệu này biến phần **Behavioral Scenarios** thành kế hoạch triển khai thực tế cho `IdlePolicy / AutonomousExplorationPolicy`, đồng thời định nghĩa rõ interface giữa các thành phần:

```text
Sensors
    ↓
Sensor Preprocessing
    ↓
Observation Builder
    ↓
IdlePolicy / TrackingPolicy
    ↓
Behavior Manager
    ↓
Safety Override
    ↓
Command Arbiter
    ↓
Robot
```

Reward chỉ được sử dụng trong training/evaluation:

```text
Environment State + Policy Action
    ↓
Reward Computation
    ↓
Training Signal
```

Nguyên tắc chính:

```text
Không train full environment ngay từ đầu.
```

Thay vào đó:

```text
Motor Sanity
    ↓
Visual Exploration
    ↓
Object Curiosity
    ↓
Dynamic Object
    ↓
Acoustic Event
    ↓
Human Presence
    ↓
Attention Seeking
    ↓
Mixed Environment
```

Mỗi scenario chỉ thêm một nhóm độ khó mới\.

---

# 2\. Kiến trúc Runtime và Interface

## 2\.1\. Sơ đồ tổng thể

```text
┌──────────────────────┐
│ Camera / IMU / Audio │
│ Encoders / Proximity │
└──────────┬───────────┘
           │ RawSensorPacket
           ▼
┌────────────────────────────┐
│ Sensor Preprocessing       │
│ - filtering                │
│ - normalization            │
│ - detection                │
│ - temporal features        │
│ - confidence estimation    │
└──────────┬─────────────────┘
           │ ProcessedObservation
           ▼
┌────────────────────────────┐
│ Behavior Manager            │
│ - mode selection            │
│ - policy scheduling         │
│ - handoff arbitration       │
│ - cooldown / hysteresis     │
└───────┬───────────┬────────┘
        │           │
        │           ├──────────────────────┐
        │           │                      │
        ▼           ▼                      ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ IdlePolicy   │ │ TrackingPolicy│ │ Other Policies │
│ explore      │ │ follow/track  │ │ future         │
└──────┬───────┘ └──────┬───────┘ └───────┬────────┘
       │ PolicyCommand  │ PolicyCommand   │
       └──────────┬─────┴──────────────────┘
                  ▼
        ┌────────────────────┐
        │ Safety Override    │
        │ - collision stop   │
        │ - limit clamp      │
        │ - emergency brake  │
        │ - command timeout  │
        └─────────┬──────────┘
                  │ SafeCommand
                  ▼
        ┌────────────────────┐
        │ Command Arbiter    │
        │ / Motor Interface  │
        └─────────┬──────────┘
                  ▼
              Robot Actuators
```

## 2\.2\. Interface giữa Sensor Preprocessing và Policy

`Sensor Preprocessing` không phát hành raw sensor trực tiếp cho policy\. Thành phần này chịu trách nhiệm:

```text
raw sensor
    ↓
timestamp alignment
    ↓
filtering
    ↓
normalization
    ↓
feature extraction
    ↓
confidence estimation
    ↓
ProcessedObservation
```

Interface đề xuất:

```python
ProcessedObservation(
    timestamp: float,
    head_pan: float,
    head_tilt: float,
    linear_velocity: float,
    angular_velocity: float,
    front_distance: float,

    current_view_signature: Vector,
    view_novelty: float,
    recent_view_history: Vector,

    object_visible: bool,
    object_x: float,
    object_y: float,
    object_size: float,
    object_confidence: float,
    object_class_embedding: Vector,
    object_novelty: float,

    motion_score: float,
    motion_bearing: float,
    scene_change_score: float,

    sound_active: bool,
    sound_level: float,
    sound_direction: Optional[float],

    face_visible: bool,
    face_x: float,
    face_y: float,
    face_size: float,
    face_confidence: float,
    human_motion: float,
    human_attention: float,

    sensor_health: SensorHealth,
)
```

Quy tắc:

```text
✓ mọi field có timestamp hoặc dùng chung frame timestamp
✓ confidence phải đi kèm detection
✓ giá trị thiếu phải có mask hoặc default rõ ràng
✓ preprocessing không quyết định behavior
✓ preprocessing không phát lệnh motor
```

## 2\.3\. Interface giữa Behavior Manager và Policy

`Behavior Manager` là thành phần quyết định policy nào được chạy và khi nào handoff xảy ra\. Policy không tự ý chuyển mode trực tiếp\.

Interface đầu vào:

```python
BehaviorContext(
    observation: ProcessedObservation,
    active_mode: BehaviorMode,
    previous_command: Optional[PolicyCommand],
    elapsed_in_mode: float,
    cooldowns: CooldownState,
    safety_state: SafetyState,
)
```

Interface đầu ra:

```python
PolicyRequest(
    policy_name: Literal[
        "idle",
        "tracking",
        "attention_seeking",
        "safe_stop",
    ],
    reason: str,
    priority: int,
    target_id: Optional[str],
    expires_at: float,
)
```

Các policy trả về:

```python
PolicyOutput(
    command: PolicyCommand,
    interest_signal: InterestSignal,
    status: PolicyStatus,
)
```

Trong đó:

```python
PolicyCommand(
    linear_velocity: float,
    angular_velocity: float,
    pan_velocity: float,
    tilt_velocity: float,
    attention_intensity: float,
    valid_for: float,
)
```

```python
InterestSignal(
    active: bool,
    type: Literal[
        "none",
        "dynamic_object",
        "human",
        "acoustic_event",
        "novel_object",
        "attention_needed",
    ],
    score: float,
    target_bearing: Optional[float],
    target_id: Optional[str],
    ttl: float,
)
```

`Behavior Manager` sử dụng `InterestSignal` để quyết định:

```text
IdlePolicy
    ↓ interest_score thấp
tiếp tục exploration

IdlePolicy
    ↓ interest_score cao + dynamic object
handoff → TrackingPolicy

IdlePolicy
    ↓ human detected
giữ IdlePolicy hoặc chuyển Social/Attention mode

IdlePolicy
    ↓ acoustic event đủ mạnh
orient / inspect ngắn hạn

bất kỳ policy nào
    ↓ safety violation
handoff → Safety Override
```

## 2\.4\. Interface giữa IdlePolicy và TrackingPolicy

`IdlePolicy` chịu trách nhiệm:

```text
explore
scan
orient
inspect ngắn hạn
phát hiện interest
return về idle
```

`TrackingPolicy` chịu trách nhiệm:

```text
duy trì target
dự đoán target motion
giữ target trong vùng quan sát
kết thúc tracking khi target mất hoặc timeout
```

Handoff request:

```python
TrackingRequest(
    target_id: str,
    target_type: Literal["dynamic_object", "human", "object"],
    initial_bearing: float,
    confidence: float,
    max_duration: float,
    handoff_reason: str,
)
```

TrackingPolicy phải trả về:

```python
TrackingStatus(
    active: bool,
    target_visible: bool,
    target_confidence: float,
    elapsed_time: float,
    lost_target_time: float,
    should_release: bool,
)
```

Quy tắc handoff:

```text
✓ IdlePolicy không chase target vô hạn
✓ TrackingPolicy phải có timeout
✓ target mất quá lâu thì release
✓ release phải trả về IdlePolicy
✓ Safety Override có quyền ngắt cả hai policy
```

## 2\.5\. Interface giữa Policy và Safety Override

Safety Override chạy sau policy output và trước motor command\.

```text
PolicyCommand
    ↓
Safety Override
    ↓
SafeCommand
```

Safety Override kiểm tra:

```text
collision risk
front distance
joint limits
velocity limits
acceleration limits
servo saturation
command timeout
sensor health
emergency stop
```

Interface:

```python
SafeCommand(
    linear_velocity: float,
    angular_velocity: float,
    pan_velocity: float,
    tilt_velocity: float,
    attention_intensity: float,
    overridden: bool,
    override_reason: Optional[str],
    safety_level: Literal["normal", "limited", "stop"],
)
```

Nguyên tắc:

```text
✓ Safety Override không tạo curiosity reward
✓ Safety Override không sửa policy state
✓ Safety Override có thể clamp hoặc stop command
✓ mọi override phải được log
✓ override rate là metric bắt buộc
```

## 2\.6\. Interface của Reward Computation

Reward computation chỉ dùng trong simulator/training pipeline, không nằm trên đường điều khiển runtime\.

```text
Environment State
        +
ProcessedObservation
        +
Policy Action
        +
Safety Events
        ↓
Reward Computation
        ↓
RewardBreakdown
```

Interface:

```python
RewardBreakdown(
    total: float,
    smooth_motion: float,
    stillness: float,
    view_novelty: float,
    object_curiosity: float,
    dynamic_response: float,
    acoustic_response: float,
    human_response: float,
    attention_success: float,
    energy_cost: float,
    repetition_penalty: float,
    collision_penalty: float,
    safety_override_penalty: float,
)
```

Reward computation phải:

```text
✓ log từng component
✓ giới hạn scale của từng component
✓ hỗ trợ ablation
✓ phát hiện reward domination
✓ không phụ thuộc vào hidden state không có trong observation
```

---

# 3\. Quy trình chuẩn cho mỗi Scenario

Mỗi scenario phải đi qua cùng một pipeline:

```text
Define Observation
        ↓
Define Environment / Events
        ↓
Define Reward
        ↓
Define Interfaces
        ↓
Random-action Validation
        ↓
Train
        ↓
Evaluate Metrics
        ↓
Check Reward Hacking
        ↓
Validate Handoff / Safety
        ↓
Accept / Revise
```

Không chuyển sang scenario tiếp theo nếu scenario hiện tại chưa đạt acceptance criteria\.

---

# 4\. Version Mapping

```text
IdlePolicy v0.1 = S0 + S1
IdlePolicy v0.2 = + S2
IdlePolicy v0.3 = + S3 + S4
IdlePolicy v0.4 = + S5 + S6
IdlePolicy v1.0 = S7 + Domain Randomization + Shadow Mode
```

---

# 5\. S0 — Motor Sanity

## Goal

Policy cần học:

```text
smooth motion
stable head movement
low jerk
respect limits
can remain still
```

## Environment

```text
empty room
```

Không có human, object, sound event hay novelty reward\.

## Observation

```text
head_pan
head_tilt
linear_velocity
angular_velocity
front_distance
```

Có thể thêm:

```text
left_motor_velocity
right_motor_velocity
```

## Action

```text
linear_velocity
angular_velocity
pan_velocity
tilt_velocity
```

Chưa cần `attention_intensity`\.

## Reward

Positive:

```text
smooth action
stable motion
appropriate stillness
```

Penalty:

```text
collision
near obstacle
jerk
servo saturation
excessive movement
safety override
```

## Interface

```text
Sensor Preprocessing
    → head state, velocity, distance

IdlePolicy
    → low-level motion command

Safety Override
    → clamp velocity, stop near obstacle

Reward Computation
    → smoothness, stillness, collision, override penalty
```

## Metrics

```text
collision_rate
motion_jerk
servo_saturation_rate
stationary_ratio
safety_override_rate
```

## Failure Modes

```text
left-right oscillation
constant movement
servo saturation
safety override loop
```

## Acceptance Criteria

```text
✓ random-action environment stable
✓ trained policy không jitter rõ rệt
✓ collision rate thấp
✓ servo saturation thấp
✓ policy có thể giữ trạng thái gần đứng yên
✓ safety override không bị kích hoạt liên tục
```

---

# 6\. S1 — Visual Exploration

## Goal

Robot học:

```text
scan
change viewpoint
reorient
observe
```

mà chưa cần object semantics\.

## Environment

Phòng có simple static geometry và nhiều vùng quan sát\.

## Observation

S0 \+

```text
current_view_signature
view_novelty
recent_view_history
```

Không cần raw RGB\.

## Reward

Positive:

```text
new_view
view_diversity
new_region
```

Penalty:

```text
repeat_view
head_oscillation
spin_forever
energy_cost
collision
safety override
```

## Interface

```text
Sensor Preprocessing
    → view signature, novelty, history

IdlePolicy
    → scan/reorient command

Behavior Manager
    → giữ IdlePolicy vì chưa có target interest

Safety Override
    → giới hạn tốc độ quay và tránh obstacle

Reward Computation
    → view diversity, repetition, energy, safety
```

## Metrics

```text
view_diversity
scene_coverage
pan_coverage
base_orientation_coverage
repetition_rate
```

## Failure Modes

```text
spin forever
head sweep exploit
hyperactivity
```

## Acceptance Criteria

```text
✓ quan sát nhiều hướng
✓ không quay base liên tục
✓ không head oscillation vô hạn
✓ có stationary period hợp lý
✓ coverage tăng theo episode
```

---

# 7\. S2 — Object Curiosity

## Goal

Expected sequence:

```text
new object
    ↓
orient
    ↓
inspect
    ↓
interest decay
    ↓
move on
```

## Environment

Random spawn:

```text
ball
cube
chair
toy
unknown object
```

Random position, orientation, class và distance\.

## Observation

S1 \+

```text
object_visible
object_x
object_y
object_size
object_confidence
object_class_embedding
object_novelty
```

## Curiosity

V1 dùng **Episodic Novelty**\. Chưa dùng RND/ICM\.

## Reward

Positive:

```text
novel object detected
orient toward novel object
inspect from useful view
new object viewpoint
```

Reward decay với cùng object/viewpoint và long stare\.

Penalty:

```text
repeat inspect
stare forever
motion jerk
safety override
```

## Interface

```text
Sensor Preprocessing
    → object detection, confidence, embedding, novelty

IdlePolicy
    → orient và inspect ngắn hạn

Behavior Manager
    → theo dõi interest decay
    → giữ IdlePolicy hoặc chuyển policy phù hợp

Safety Override
    → giới hạn motion khi object ở gần obstacle

Reward Computation
    → novelty, useful inspection, decay, repetition
```

## Metrics

```text
object_discovery_count
novel_object_inspection_rate
mean_inspection_time
object_revisit_rate
novelty_reward_fraction
```

## Failure Modes

```text
stare forever
object switching too fast
camera jitter novelty exploit
```

## Acceptance Criteria

```text
✓ ưu tiên object mới
✓ inspect trong khoảng hợp lý
✓ interest giảm theo thời gian
✓ chuyển target khác
✓ không exploit camera jitter
```

---

# 8\. S3 — Dynamic Object

## Goal

```text
motion detected
      ↓
orient
      ↓
observe
      ↓
hold / disengage
```

## Environment

```text
ball rolling
moving toy
object crossing FOV
```

Random start time, direction, speed và distance\.

## Observation

S2 \+

```text
motion_score
motion_bearing
scene_change_score
```

## Reward

Positive:

```text
orient toward motion
observe dynamic event
```

Penalty:

```text
chase forever
continuous rotation
false-trigger response
safety override
```

## Policy Boundary

IdlePolicy chỉ orient \+ observe\. Nếu `object_interest` vượt threshold, Behavior Manager có thể handoff sang `TrackingPolicy`\.

```text
IdlePolicy
    ↓ InterestSignal(type="dynamic_object")
Behavior Manager
    ↓ TrackingRequest
TrackingPolicy
    ↓ TrackingStatus
Behavior Manager
    ↓ release hoặc return IdlePolicy
```

## Interface

```text
Sensor Preprocessing
    → motion_score, motion_bearing, target confidence

IdlePolicy
    → orient và observe

Behavior Manager
    → threshold, hysteresis, timeout, handoff

TrackingPolicy
    → duy trì target trong thời gian giới hạn

Safety Override
    → ngắt tracking nếu có collision risk

Reward Computation
    → response latency, orientation, false response, chase penalty
```

## Metrics

```text
motion_response_latency
orientation_accuracy
false_response_rate
handoff_rate
tracking_timeout_rate
```

## Acceptance Criteria

```text
✓ phản ứng khi có movement
✓ latency ổn định
✓ không chase vô hạn
✓ biết return về idle
✓ tạo được interest signal cho Behavior Manager
✓ handoff không gây command discontinuity
```

---

# 9\. S4 — Acoustic Event

## Goal

Robot học phản ứng với acoustic event\.

## Observation

Nếu chỉ có một INMP441:

```text
sound_active
sound_level
```

Nếu có microphone array/DOA:

```text
sound_direction
```

## Event Types

```text
short sound
repeated sound
loud transient
human-like event
background noise
```

## Reward

Positive:

```text
appropriate orientation response
brief inspection
```

Penalty:

```text
respond to every noise
continuous movement
repeat response
safety override
```

## Interface

```text
Sensor Preprocessing
    → sound activity, level, direction, event confidence

IdlePolicy
    → orient hoặc inspect ngắn

Behavior Manager
    → áp dụng threshold, cooldown và refractory period

Safety Override
    → giới hạn response trong môi trường có obstacle

Reward Computation
    → response quality, false response, repeat response
```

## Metrics

```text
sound_response_rate
false_response_rate
response_latency
repeat_response_rate
```

## Failure Modes

```text
noise hypersensitivity
event loop
```

## Acceptance Criteria

```text
✓ phản ứng với event đủ mạnh
✓ bỏ qua background noise phần lớn thời gian
✓ response không kéo dài vô hạn
✓ return to idle
```

---

# 10\. S5 — Human Presence

## Goal

Thêm social gaze cơ bản, chưa full social interaction\.

Expected sequence:

```text
human appears
    ↓
brief gaze
    ↓
observe
    ↓
look away
    ↓
possible revisit
```

## Environment

Human abstraction với random position, distance, movement và attention state\.

## Observation

```text
face_visible
face_x
face_y
face_size
face_confidence
human_motion
human_attention
```

## Reward

Positive:

```text
brief human orientation
mutual attention event
appropriate revisit
```

Continuous gaze phải bị reward decay/saturation\.

## Interface

```text
Sensor Preprocessing
    → face detection, confidence, motion, attention estimate

IdlePolicy
    → orient và gaze ngắn

Behavior Manager
    → quản lý social priority, gaze timeout, revisit cooldown

TrackingPolicy
    → chỉ được kích hoạt nếu human tracking được cho phép ở mode hiện tại

Safety Override
    → ưu tiên khoảng cách an toàn và giới hạn motion

Reward Computation
    → human orientation, mutual attention, stare penalty
```

## Metrics

```text
human_orientation_count
mean_gaze_duration
gaze_revisit_rate
stare_rate
```

## Failure Modes

```text
stare forever
ignore human
social obsession
```

## Acceptance Criteria

```text
✓ orient về human
✓ gaze duration hợp lý
✓ có look-away
✓ có revisit khi phù hợp
✓ không stare vô hạn
```

---

# 11\. S6 — Attention Seeking

## Goal

Robot học cách thử gây chú ý với **chi phí tín hiệu thấp nhất có hiệu quả**\.

## Action Extension

Thêm:

```text
attention_intensity
```

## Expression Mapping

```text
0.0–0.2  nothing
0.2–0.4  micro movement
0.4–0.6  gaze / head gesture
0.6–0.8  stronger gesture
0.8–1.0  audio/LED if cooldown allows
```

## Human Agent

HumanAgent trả:

```text
human_attention_response
```

Probability được random hóa giữa episode\.

## Reward

```text
+ human_response
- signal_cost
- repeated_signal
- spam
- safety override
```

Desired escalation:

```text
look
 ↓
wait
 ↓
gesture
 ↓
wait
 ↓
stronger signal only if needed
```

## Interface

```text
Behavior Manager
    → tạo AttentionRequest khi policy cần social response

IdlePolicy / Attention Policy
    → chọn intensity và thời điểm phát tín hiệu

Sensor Preprocessing
    → phát hiện human response và attention state

Safety Override
    → áp dụng audio/LED cooldown, giới hạn gesture và motor

Reward Computation
    → success, signal cost, escalation, spam
```

## Metrics

```text
attention_attempt_count
attention_success_rate
signal_cost_per_success
signal_spam_rate
mean_escalation_level
```

## Failure Modes

```text
beep spam
immediate strongest signal
infinite escalation
```

## Acceptance Criteria

```text
✓ có attention attempt
✓ ưu tiên low-cost signal trước
✓ biết chờ response
✓ escalation có kiểm soát
✓ spam rate thấp
✓ biết dừng khi không hiệu quả
```

---

# 12\. S7 — Mixed Environment

## Goal

Integration scenario gồm:

```text
human
static objects
dynamic object
sound
obstacles
unknown object
```

Policy phải tự phân bổ attention theo context\.

## Environment Randomization

```text
room layout
object count
object classes
human state
event timing
event overlap
sensor noise
motor parameters
```

## Reward

Dùng reward tổng đã được tuning từ S0–S6\. Không thêm quá nhiều reward component mới ở S7\.

## Runtime Arbitration

```text
ProcessedObservation
        ↓
Behavior Manager
        ├── IdlePolicy
        ├── TrackingPolicy
        ├── Attention Policy
        └── Safe Stop
        ↓
Safety Override
        ↓
SafeCommand
```

Behavior Manager phải xử lý:

```text
dynamic object + human cùng lúc
sound event trong khi tracking
obstacle xuất hiện trong lúc orient
attention request trong quiet period
target mất sau handoff
```

## Metrics

```text
scene_coverage
view_diversity
object_discovery
behavior_entropy
repetition_rate
collision_rate
energy_proxy
human_orientation
attention_success
signal_spam
safety_override_rate
handoff_rate
policy_switch_rate
```

## Failure Modes

```text
reward domination
social domination
curiosity domination
hyperactivity
behavioral attractor
handoff thrashing
safety override loop
```

## Acceptance Criteria

```text
✓ behavior đa dạng
✓ response phụ thuộc context
✓ có quiet period
✓ không infinite loop
✓ low collision
✓ low spam
✓ policy handoff hoạt động
✓ không có reward component domination rõ rệt
✓ không chuyển policy liên tục vì nhiễu
✓ Safety Override luôn có quyền ưu tiên cao nhất
```

---

# 13\. Scenario Test Matrix

|Scenario|Main Test              |
|--------|-----------------------|
|S0      |Motion stability       |
|S1      |View exploration       |
|S2      |Curiosity              |
|S3      |Dynamic response       |
|S4      |Acoustic response      |
|S5      |Human gaze             |
|S6      |Attention seeking      |
|S7      |Multi-event arbitration|

Stress tests bổ sung:

```text
high noise
motor mismatch
servo delay
sensor latency
dense obstacles
long horizon
simultaneous events
policy handoff delay
safety override activation
```

---

# 14\. Scenario Progression Rules

Chỉ unlock scenario mới khi:

```text
required metrics pass
AND
no critical reward exploit
AND
behavior reproducible across multiple seeds
AND
interface contract pass
AND
handoff không gây instability
AND
safety override hoạt động đúng
```

Khuyến nghị tối thiểu:

```text
3 seeds
```

cho mỗi checkpoint quan trọng\.

Mỗi checkpoint phải kiểm tra:

```text
observation schema
action schema
reward breakdown
policy output
Behavior Manager transition
Safety Override log
```

---

# 15\. Recommended First Implementation Sprints

## Sprint A — S0

Tạo:

```text
simulation/assets/robot/trax_base.xml
simulation/worlds/empty_room.xml
simulation/envs/trax_idle_env.py
```

Action:

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

Thêm interface:

```text
ProcessedObservation
PolicyCommand
SafeCommand
SafetyState
RewardBreakdown
```

Mục tiêu:

```text
random-action simulation stable
```

## Sprint B — S1

Thêm:

```text
view signature
view memory
view diversity reward
quiet reward
repetition penalty
```

Train SAC baseline\.

Mục tiêu:

```text
robot biết scan và biết dừng
```

## Sprint C — S2

Thêm:

```text
semantic object
episodic novelty
object curiosity
novelty decay
interest signal
```

Mục tiêu:

```text
object mới → inspect → mất interest → move on
```

## Sprint D — S3

Thêm:

```text
motion event
dynamic target
TrackingRequest
TrackingStatus
handoff timeout
```

Mục tiêu:

```text
dynamic object → orient → optional tracking → release → idle
```

## Sprint E — Safety and Arbitration

Thêm:

```text
Behavior Manager
Safety Override
command arbitration
override logging
policy transition metrics
```

Mục tiêu:

```text
mọi policy đều chạy qua cùng một safety boundary
```

---

# 16\. Implementation Principle

Thứ tự ưu tiên:

```text
stable
    ↓
safe
    ↓
non-repetitive
    ↓
curious
    ↓
responsive
    ↓
social
    ↓
natural
```

Nếu policy chưa ổn định ở S0/S1 thì không nên thêm social complexity của S5/S6\.

Các nguyên tắc interface:

```text
✓ Sensor Preprocessing chỉ tạo observation
✓ Reward Computation chỉ tạo training signal
✓ Behavior Manager quyết định policy mode
✓ Policy chỉ đề xuất command
✓ Safety Override có quyền cuối cùng trước actuator
✓ TrackingPolicy không tự giữ quyền điều khiển vô hạn
✓ mọi handoff phải có timeout và release condition
✓ mọi safety override phải được đo và log
```

---

# 17\. Definition of Done

Behavioral Scenario pipeline hoàn thành khi S0–S7 đều có:

```text
environment
observation
action
reward
metrics
failure-mode tests
acceptance criteria
interface contract
handoff rules
safety behavior
```

và `IdlePolicy v1.0` hoạt động trong Mixed Environment với:

```text
low collision
low repetition
reasonable energy
novelty response
human awareness
controlled attention seeking
stable policy handoff
bounded tracking duration
low safety override rate
reproducible behavior across seeds
```

Sơ đồ runtime cuối cùng phải được kiểm chứng theo chuỗi:

```text
sensor
    ↓
sensor preprocessing
    ↓
processed observation
    ↓
Behavior Manager
    ↓
IdlePolicy / TrackingPolicy
    ↓
policy command
    ↓
Safety Override
    ↓
safe command
    ↓
actuator
```

Sơ đồ training phải được kiểm chứng theo chuỗi:

```text
environment state
    +
processed observation
    +
policy action
    +
safety events
    ↓
reward computation
    ↓
reward breakdown
    ↓
policy update
```