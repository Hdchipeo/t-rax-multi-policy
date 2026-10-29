# 01 — System Architecture

## Mục tiêu

Định nghĩa ranh giới giữa Perception, World State, Behavior Manager, RL Policy, Safety Layer và low-level controller.

## Kiến trúc

```text
Camera ─► Vision ──────┐
ToF ───────────────────┤
Audio ─────────────────┤
Robot State ───────────┤
                       ▼
                  World State
                       │
                       ▼
               Behavior Manager
                       │
                       ▼
                Policy Selector
                       │
       ┌───────────────┼──────────────┐
       ▼               ▼              ▼
   IdlePolicy     TrackingPolicy   BallPolicy
       │
       ▼
     Action
       │
       ▼
  Safety Layer
       │
       ▼
 Controllers
       │
       ▼
  Actuators
```

## Behavior Manager

Quyết định:
- policy active;
- priority;
- interrupt;
- handoff;
- cooldown;
- minimum active time.

Ví dụ:

```cpp
Behavior selectBehavior(const Context& ctx)
{
    if (ctx.emergency) return Behavior::SAFE_STOP;
    if (ctx.user_command_active) return Behavior::COMMAND;
    if (ctx.human_engagement > 0.80f) return Behavior::SOCIAL;
    if (ctx.ball_interest > 0.75f) return Behavior::BALL;
    return Behavior::IDLE;
}
```

## IdlePolicy

Tên kỹ thuật phù hợp hơn: `AutonomousExplorationPolicy`.

Trách nhiệm:
- gaze;
- light exploration;
- novelty response;
- object inspection;
- attention seeking mức thấp;
- tạo interest signal.

Không chịu trách nhiệm:
- emergency;
- battery protection;
- hard obstacle stop;
- policy priority.

## Internal Drives

```text
curiosity
boredom
social_need
energy_budget
perception_confidence
```

## Interface

```text
Observation → Policy → RobotAction → Safety → Controller
```

Policy không truy cập trực tiếp hardware driver.

## Definition of Done

- policy thay thế độc lập được;
- simulator và firmware dùng cùng semantic observation;
- Behavior Manager override được;
- Safety luôn đứng sau policy;
- schema có version.
