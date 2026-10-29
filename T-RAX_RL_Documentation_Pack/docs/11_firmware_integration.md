# 11 — ESP-IDF Firmware Integration

## Architecture

```text
Perception Tasks
      ↓
World State
      ↓
Observation Adapter
      ↓
Policy Runtime
      ↓
Safety Supervisor
      ↓
Motion Controller
      ├─ Motor HAL
      └─ Servo HAL
```

## Observation

```cpp
struct IdleObservation {
    float robot[8];
    float vision[16];
    float temporal[6];
    float internal[5];
    float exploration[4];
};
```

## Action

```cpp
struct IdleAction {
    float linear;
    float angular;
    float pan_velocity;
    float tilt_velocity;
    float attention;
};
```

## Policy Runner

```cpp
class IdlePolicyRunner {
public:
    bool init();
    IdleAction infer(const IdleObservation& observation);
};
```

Policy runtime không gọi trực tiếp LEDC/MCPWM/GPIO.

## Scheduling baseline

```text
perception          variable
world state         20–50 Hz
policy inference     5–10 Hz
motion controller   50–100 Hz
```

## Metadata

Verify trước khi load:

```text
policy_id
policy_version
observation_schema
action_schema
quantization
input_shape
output_shape
```

Mismatch → reject model.

## Failure fallback

```text
inference error
   ↓
disable RL policy
   ↓
deterministic idle controller
```

Không làm mất control loop.

## Telemetry

```text
timestamp
active_policy
policy_version
observation_summary
raw_action
safe_action
safety_override_reason
curiosity
boredom
social_need
behavior_handoff
```
