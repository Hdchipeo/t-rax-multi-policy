# T-Rax Autonomous Behavioral Dynamics

Bộ tài liệu implementation cho hệ thống **Autonomous Behavioral Dynamics** của robot T-Rax.

## Kiến trúc mục tiêu

```text
Sensors / Perception
        ↓
World State
        ↓
Behavior Manager
        ↓
Policy Selector
        ↓
IdlePolicy
        ↓
Safety Layer
        ↓
Robot
```

`IdlePolicy` chỉ tối ưu hành vi nền. Behavior Manager giữ quyền chọn policy và Safety Layer luôn có quyền override.

## Bộ tài liệu

1. `01_system_architecture.md`
2. `02_mujoco_environment.md`
3. `03_observation_action_spec.md`
4. `04_reward_design.md`
5. `05_curiosity_novelty.md`
6. `06_social_attention.md`
7. `07_behavior_manager_safety.md`
8. `08_training_pipeline.md`
9. `09_evaluation_testing.md`
10. `10_sim_to_real.md`
11. `11_firmware_integration.md`
12. `12_implementation_roadmap.md`

## Thứ tự triển khai

```text
Architecture
   ↓
MuJoCo model
   ↓
Gymnasium env
   ↓
Observation / Action
   ↓
Reward V1
   ↓
SAC baseline
   ↓
Curiosity
   ↓
Human / Social
   ↓
Evaluation
   ↓
Domain Randomization
   ↓
Shadow Mode
   ↓
ESP-IDF deployment
```

## Không dùng RL cho

```text
emergency stop
hard collision stop
servo hard limits
battery protection
watchdog
user command priority
communication failsafe
```

## Stack

```text
MuJoCo
Gymnasium
Stable-Baselines3
PyTorch
TensorBoard
ONNX
ESP-IDF
```
