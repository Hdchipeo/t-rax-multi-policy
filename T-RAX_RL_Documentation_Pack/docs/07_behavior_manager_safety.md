# 07 — Behavior Manager & Safety

## Priority baseline

```text
Emergency
  >
User Command
  >
Safety Recovery
  >
Explicit Task
  >
Social / Tracking / Ball
  >
Idle
```

## Behavior enum

```cpp
enum class Behavior {
    SAFE_STOP,
    COMMAND,
    RECOVERY,
    SOCIAL,
    TRACK,
    BALL,
    IDLE,
};
```

## Hysteresis

Dùng:
```text
enter_threshold
exit_threshold
minimum_duration
cooldown
```

Không switch policy trên một sample.

## Safety Layer

Deterministic:
- velocity clamp;
- angular velocity clamp;
- servo angle/rate clamp;
- obstacle hard stop;
- battery protection;
- emergency stop;
- watchdog.

## C++ skeleton

```cpp
struct RobotAction {
    float linear;
    float angular;
    float pan_velocity;
    float tilt_velocity;
    float attention;
};
```

Safety phải biến:

```text
raw_action → safe_action
```

## Telemetry

```text
raw_action
safe_action
override_reason
override_count
```

Metric:

```text
safety_override_rate
```

Policy tốt phải giảm dependency vào safety override theo thời gian.
