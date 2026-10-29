# 06 — Social Attention

## Mục tiêu

Cho IdlePolicy có social behavior mức thấp, nhưng không biến nó thành full SocialPolicy.

## Social observation

```text
face_visible
face_x
face_y
face_size
face_confidence
human_attention
human_motion
time_since_human_seen
time_since_human_response
```

## Desired sequences

```text
human appears
   ↓
brief gaze
   ↓
observe
   ↓
look away
   ↓
revisit
```

Attention seeking:

```text
look
 ↓
wait
 ↓
head gesture
 ↓
wait
 ↓
stronger signal
```

## Attention intensity

Policy output:

```text
attention_intensity ∈ [0,1]
```

Expression Layer:

```text
0.0–0.2  nothing
0.2–0.4  micro head motion
0.4–0.6  gaze + head gesture
0.6–0.8  stronger gesture
0.8–1.0  audio/LED if cooldown allows
```

## Human Agent V1

```python
class HumanAgent:
    def update(self, robot):
        p = 0.02
        if robot.looking_at_human: p += 0.10
        if robot.head_gesture: p += 0.15
        if robot.body_orientation: p += 0.10
        if robot.audio_signal: p += 0.30
        if robot.signal_spam: p -= 0.40
        return np.random.random() < np.clip(p, 0.0, 1.0)
```

## Randomize

```text
reaction_latency
gaze_sensitivity
motion_sensitivity
sound_sensitivity
preferred_distance
response_probability
```

## Gaze saturation

Không reward stare liên tục.

## Handoff

Khi human interest đủ cao và kéo dài đủ lâu:

```text
IdlePolicy → SocialPolicy
```
