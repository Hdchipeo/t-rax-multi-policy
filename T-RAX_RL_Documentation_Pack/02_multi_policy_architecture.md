# Kiến trúc Multi-Policy cho T-RAX

## 1. Mục tiêu thiết kế

Kiến trúc phải đáp ứng:

- policy độc lập;
- shared perception;
- shared safety;
- unified observation;
- unified logging;
- policy hot-switch;
- version hóa;
- khả năng export embedded;
- tương thích policy mới mà không sửa core.

---

## 2. Kiến trúc tổng thể

```text
                           User / LLM
                               │
                         High-Level Intent
                               │
                               ▼
                       Behavior Manager
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      AutonomousIdle      FollowPerson       PlayBall
             │                 │                 │
             ├────────── SearchObject ──────────┤
             │                                   │
             └────────── Explore / Dock ─────────┘
                               │
                               ▼
                         Action Contract
                               │
                               ▼
                       Behavior Executor
                               │
                               ▼
                       Safety Supervisor
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              Motor          Servo         Stop
                 │             │
                 └──────┬──────┘
                        ▼
                    Hardware
```

---

## 3. Shared World State

Không để mỗi policy tự đọc trực tiếp hardware.

Tạo một lớp trung gian:

```cpp
struct WorldState {
    uint64_t timestamp_ms;

    // Perception
    bool person_visible;
    float person_x;
    float person_y;
    float person_distance;

    bool ball_visible;
    float ball_x;
    float ball_y;
    float ball_distance;

    // Environment
    float obstacle_front;

    // Robot
    float head_pan;
    float head_tilt;
    float linear_velocity;
    float angular_velocity;

    // History
    float time_since_person_seen;
    float time_since_ball_seen;
    uint8_t previous_action;
};
```

Policy adapter chỉ lấy feature cần thiết.

---

## 4. Observation Adapter

```text
WorldState
    │
    ├── IdleObservationAdapter
    ├── FollowObservationAdapter
    ├── BallObservationAdapter
    └── ExploreObservationAdapter
```

Ví dụ:

```cpp
struct IdleObservation {
    float person_visible;
    float person_x;
    float motion_level;
    float head_pan;
    float time_since_event;
    float previous_action;
};
```

Lợi ích:

- policy không phụ thuộc struct lớn;
- version schema riêng;
- dễ unit test;
- dễ simulate.

---

## 5. Action Contract

Không để output neural network gọi driver trực tiếp.

```cpp
enum class PrimitiveAction : uint8_t {
    REST,
    LOOK_LEFT,
    LOOK_RIGHT,
    LOOK_CENTER,
    TRACK_TARGET,
    TURN_LEFT,
    TURN_RIGHT,
    MOVE_FORWARD_SHORT,
    MOVE_BACKWARD_SHORT
};
```

Behavior Executor:

```text
RL output
  ↓
PrimitiveAction
  ↓
Behavior Executor
  ↓
trajectory / velocity request
  ↓
Safety Supervisor
  ↓
driver
```

---

## 6. Behavior Interface

```cpp
class IBehavior {
public:
    virtual ~IBehavior() = default;

    virtual void enter(const WorldState& state) = 0;
    virtual void update(const WorldState& state) = 0;
    virtual void exit() = 0;
};
```

Policy behavior:

```cpp
class RLBehavior : public IBehavior {
public:
    void enter(const WorldState& state) override;
    void update(const WorldState& state) override;
    void exit() override;

private:
    PolicyRuntime policy_;
};
```

---

## 7. Priority

Đề xuất:

```text
Emergency / Safety
       >
Battery Critical
       >
User Command
       >
Active Task
       >
Autonomous Idle
```

Idle không được quyền chặn command cấp cao.

---

## 8. Policy Manifest

Mỗi policy nên có metadata:

```json
{
  "name": "autonomous_idle",
  "version": "1.0.0",
  "observation_schema": "idle_obs_v1",
  "action_schema": "idle_action_v1",
  "network": [16, 64, 64, 8],
  "algorithm": "PPO",
  "quantization": "int8",
  "control_hz": 10,
  "training_env": "TRAXIdle-v1"
}
```

Không deploy file model mà không có manifest.

---

## 9. Registry

Training:

```python
POLICIES = {
    "idle": IdlePolicyConfig,
    "follow": FollowPolicyConfig,
    "play_ball": PlayBallPolicyConfig,
}
```

Firmware:

```cpp
enum class PolicyId {
    IDLE,
    FOLLOW,
    PLAY_BALL
};
```

Mục tiêu là mapping rõ ràng giữa:

```text
training artifact
↕
manifest
↕
firmware runtime
```

---

## 10. Safety Boundary

Những thứ không được RL sở hữu:

- emergency stop;
- hard obstacle cutoff;
- motor current limit;
- battery undervoltage;
- servo mechanical limit;
- watchdog;
- motor timeout;
- invalid observation;
- model output NaN/out-of-range.

Ví dụ:

```cpp
if (!std::isfinite(action_value)) {
    emergency_stop();
}

if (world.obstacle_front < MIN_SAFE_DISTANCE) {
    motor_stop();
}
```

---

## 11. Policy Switching

Switch phải có lifecycle:

```text
current.exit()
      ↓
zero / safe command
      ↓
new.enter()
      ↓
new.update()
```

Không switch trực tiếp khi motor command cũ còn tồn tại.

---

## 12. Kiến trúc thư mục firmware

```text
components/
├── perception/
├── world_state/
├── behavior_manager/
├── policy_runtime/
├── behavior_executor/
├── safety_supervisor/
├── motor_control/
├── servo_control/
└── policies/
    ├── idle/
    ├── follow/
    └── play_ball/
```

Mỗi policy không chứa code driver.
