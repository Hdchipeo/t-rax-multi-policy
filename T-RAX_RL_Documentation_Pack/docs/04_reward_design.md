# 04 — Reward Design

## Mục tiêu

Reward phải tạo behavior:

```text
interesting
responsive
safe
non-repetitive
energy-aware
```

## Reward tổng

```text
R =
 + novelty
 + new_view
 + human_interest
 + engagement
 + appropriate_quiet
 - collision
 - near_obstacle
 - jerk
 - repetition
 - signal_spam
 - energy_cost
```

## Baseline weights

```text
novelty             +0.30
new_view            +0.15
human_interest      +0.30
engagement          +0.50
appropriate_quiet   +0.05

collision           -2.00
near_obstacle       -0.30
motion_jerk         -0.20
repetition          -0.15
signal_spam         -0.50
energy_cost         -0.05
```

Chỉ là starting point.

## Quiet reward

Robot phải học rằng không làm gì đôi khi là hành động đúng.

```python
if low_novelty and no_event and action_energy < threshold:
    reward += quiet_reward
```

## Anti reward-hacking

### Spin forever
Không reward distance traveled đơn thuần.

### Head oscillation
Dùng:
```text
temporal novelty memory
jerk penalty
repetition penalty
pose quantization
```

### Stare forever
Dùng gaze reward decay.

### Attention spam
Dùng:
```text
signal cost
cooldown
repeat penalty
human-response reward
```

## Logging

Bắt buộc log component riêng:

```text
reward/novelty
reward/view
reward/human
reward/engagement
reward/quiet
penalty/collision
penalty/jerk
penalty/repetition
penalty/signal
penalty/energy
```

## Versioning

```text
reward_v1 = smooth + exploration
reward_v2 = + novelty
reward_v3 = + human
reward_v4 = + attention
```
