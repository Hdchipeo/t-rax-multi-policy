# 09 — Evaluation & Testing

## Không chỉ nhìn cumulative reward

Reward cao vẫn có thể tương ứng behavior xấu.

## Exploration metrics

```text
scene_coverage
view_diversity
object_discovery_count
novel_object_inspection_rate
```

## Behavior metrics

```text
behavior_entropy
repetition_rate
stationary_ratio
mean_motion_segment_duration
```

## Motion metrics

```text
jerk
servo_reversal_rate
angular_oscillation
energy_proxy
```

## Social metrics

```text
human_orientation_count
attention_attempt_count
attention_success_rate
signal_spam_rate
```

## Safety metrics

```text
collision_rate
near_collision_rate
safety_override_rate
servo_saturation_rate
```

## Test matrix

```text
empty room
sparse room
dense room
objects only
human only
dynamic object
mixed environment
high sensor noise
high motor mismatch
```

## Long horizon

```text
1 h
4 h
24 h equivalent simulation
```

Tìm behavioral attractor như:

```text
left-right-left-right
rotate-wait-rotate-wait
```

Có thể dùng autocorrelation, FFT action history hoặc sequence hashing.

## Acceptance

### v0.1
- stable locomotion;
- smooth head;
- low collision;
- can stay still;
- multiple viewpoints;
- no obvious infinite loop.

### v0.2
- novelty;
- curiosity decay;
- object inspection.

### v0.3
- human orientation;
- attention attempt;
- low spam;
- policy handoff.
