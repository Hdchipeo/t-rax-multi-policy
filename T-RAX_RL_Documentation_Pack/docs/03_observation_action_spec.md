# 03 — Observation & Action Specification

## Observation V1

### Robot — 8

```text
head_pan_norm
head_tilt_norm
linear_velocity_norm
angular_velocity_norm
left_motor_velocity_norm
right_motor_velocity_norm
front_distance_norm
battery_norm
```

### Vision — 16

```text
face_visible
face_x
face_y
face_size
face_confidence

object_visible
object_x
object_y
object_size
object_confidence

motion_score
scene_change_score

object_embedding[4]
```

### Temporal — 6

```text
time_since_new_object
time_since_human_seen
time_since_human_response
time_since_motion
time_since_attention_signal
time_since_base_motion
```

### Internal — 5

```text
curiosity
boredom
social_need
energy_budget
perception_confidence
```

### Exploration — 4

```text
current_view_novelty
local_region_novelty
object_novelty
recent_action_diversity
```

Tổng: khoảng 39 scalar features.

## Gymnasium

```python
self.observation_space = gym.spaces.Dict({
    "robot": gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
    "vision": gym.spaces.Box(-1.0, 1.0, (16,), dtype=np.float32),
    "temporal": gym.spaces.Box(0.0, 1.0, (6,), dtype=np.float32),
    "internal": gym.spaces.Box(0.0, 1.0, (5,), dtype=np.float32),
    "exploration": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
})
```

## Action V1

```text
a[0] linear_velocity
a[1] angular_velocity
a[2] pan_velocity
a[3] tilt_velocity
a[4] attention_intensity
```

Policy output normalized `[-1, 1]`; attention remap về `[0,1]`.

## Không đưa vào observation

Không dùng simulator-only information như:
- future events;
- perfect human intent;
- perfect object identity nếu robot thật không có;
- ground-truth coordinate mà real robot không đo được.

## Version

```text
observation_schema = idle_obs_v1
action_schema      = idle_action_v1
```
