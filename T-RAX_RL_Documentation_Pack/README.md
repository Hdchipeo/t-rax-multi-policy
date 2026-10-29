# T-RAX RL Documentation Pack

Bộ tài liệu thiết kế và triển khai Reinforcement Learning cho robot T-RAX.

## Nội dung

1. `00_overview.md` — đặt vấn đề, mục tiêu, lý do chọn multi-policy.
2. `01_theory.md` — cơ sở lý thuyết RL cần cho dự án.
3. `02_multi_policy_architecture.md` — kiến trúc multi-policy và firmware boundary.
4. `03_training_mac_m4_mujoco.md` — hướng dẫn train trên Mac mini M4 + MuJoCo.
5. `04_scalable_policy_pipeline.md` — pipeline chuẩn hóa để mở rộng policy tương lai.
6. `05_implementation_roadmap.md` — roadmap, checklist và KPI.

## Stack đề xuất

- MuJoCo
- Gymnasium
- Stable-Baselines3
- PPO
- PyTorch
- ESP-IDF cho runtime robot

## Tài liệu chính thức nên tham khảo

- MuJoCo Python documentation
- Gymnasium custom environment documentation
- Stable-Baselines3 PPO documentation
- PyTorch MPS backend documentation

## Nguyên tắc cốt lõi

```text
LLM / Planner
      ↓
Behavior Manager
      ↓
Multi-Policy Skill Library
      ↓
Behavior Executor
      ↓
Safety Supervisor
      ↓
ESP-IDF Drivers
```

Không để RL bypass safety layer.
