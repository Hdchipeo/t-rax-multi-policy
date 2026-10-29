# 10 — Sim-to-Real

## Domain Randomization

### Physics
```text
mass
center_of_mass
friction
motor_gain
left_right_mismatch
servo_speed
servo_delay
```

### Sensors
```text
ToF_noise
vision_latency
confidence_noise
false_positive
false_negative
audio_latency
```

### Environment
```text
room_size
object_count
object_placement
event_timing
human_position
```

## Example

```python
def randomize_domain(self):
    self.motor_gain_left = np.random.uniform(0.85, 1.15)
    self.motor_gain_right = np.random.uniform(0.85, 1.15)
    self.servo_delay = np.random.uniform(0.02, 0.12)
    self.track_friction = np.random.uniform(0.70, 1.20)
    self.vision_latency = np.random.uniform(0.04, 0.20)
    self.tof_sigma = np.random.uniform(0.005, 0.020)
```

Range cuối phải dựa trên measurement robot thật.

## Pipeline

```text
Simulation
   ↓
Domain Randomization
   ↓
Offline Evaluation
   ↓
Export
   ↓
Shadow Mode
   ↓
Limited Actuation
   ↓
Real Deployment
```

## Shadow Mode

Policy nhận observation thật nhưng không điều khiển actuator.

Log:

```text
observation
policy_action
safety_filtered_action
actual_controller_action
```

## Limited actuation

Triển khai theo thứ tự:

```text
head only
 ↓
body rotation
 ↓
low-speed base
 ↓
full bounded idle behavior
```
