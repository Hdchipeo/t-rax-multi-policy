# 05 — Curiosity & Novelty

## Nguyên tắc

Curiosity phải thưởng cho **information gain / novelty**, không thưởng cho activity thuần túy.

## V1 — Episodic Novelty

Memory:

```text
M = {z1, z2, ..., zn}
```

Compact state có thể gồm:

```text
coarse robot pose
head pan
head tilt
visible object classes
primary object bearing
scene signature
```

Novelty:

```text
distance = nearest_distance(current_state, episodic_memory)
```

Reward tăng khi distance lớn.

## Memory insertion

```python
if nearest_distance > novelty_threshold:
    memory.append(z)
```

Không lưu mọi frame.

Baseline memory:

```text
256–2048 entries / episode
```

## Novelty decay

Object mới có interest cao, sau đó decay khi inspect lâu.

## V2 — RND

```text
observation
   ├─► fixed random target
   └─► predictor

prediction error
      ↓
intrinsic reward
```

## V3 — ICM

```text
state + action
      ↓
forward model
      ↓
predicted next feature

actual next feature
      ↓
prediction error
      ↓
intrinsic reward
```

## Roadmap

```text
Episodic Novelty
       ↓
RND benchmark
       ↓
ICM benchmark
```

Không bắt đầu bằng ICM.

## Metrics

```text
mean_novelty
novel_states_per_episode
memory_size
repeat_visit_count
novelty_reward_fraction
object_revisit_frequency
```
