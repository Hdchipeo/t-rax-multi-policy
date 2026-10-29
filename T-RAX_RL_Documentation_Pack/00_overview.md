# T-RAX Reinforcement Learning — Tổng quan và đặt vấn đề

## 1. Bối cảnh

T-RAX hiện sử dụng các kịch bản IDLE định trước. Cách này dễ triển khai nhưng tạo ra ba giới hạn chính:

1. Hành vi lặp lại và dễ đoán.
2. Robot phản ứng yếu với thay đổi của môi trường.
3. Mỗi hành vi mới cần viết thêm logic điều kiện, làm hệ thống ngày càng khó mở rộng.

Mục tiêu của hướng RL không phải là thay toàn bộ firmware bằng AI, mà là tạo một tầng **Behavior Policy** có khả năng lựa chọn hành vi dựa trên trạng thái hiện tại, trong khi safety, driver và các giới hạn vật lý vẫn do ESP-IDF kiểm soát xác định.

---

## 2. Mục tiêu hệ thống

Kiến trúc đề xuất:

```text
Perception
  │
  ├── YOLO / vision
  ├── ToF
  ├── IMU
  └── history
       │
       ▼
World State
       │
       ▼
Behavior Manager
       │
       ├── AutonomousIdlePolicy
       ├── FollowPersonPolicy
       ├── PlayBallPolicy
       ├── SearchObjectPolicy
       ├── ExplorePolicy
       └── DockingPolicy
               │
               ▼
        Behavior Executor
               │
               ▼
        Safety Controller
               │
               ▼
        ESP-IDF Drivers
```

### Nguyên tắc

- RL không trực tiếp sở hữu emergency stop.
- RL không điều khiển PWM thô trong giai đoạn đầu.
- Mỗi policy chỉ giải quyết một kỹ năng.
- Các policy dùng chung schema và toolchain.
- LLM chỉ chọn intent/skill cấp cao, không điều khiển motor từng frame.
- Policy phải chạy local sau khi deploy.
- Training chạy trên Mac mini M4; ESP32-S3/P4 chỉ inference.

---

## 3. Tại sao chọn Multi-Policy

### Multi-policy

```text
IdlePolicy
FollowPolicy
PlayBallPolicy
SearchPolicy
ExplorePolicy
```

Ưu điểm:

- Reward của từng behavior rõ ràng.
- Dễ test độc lập.
- Dễ rollback.
- Dễ version.
- Dễ tối ưu model riêng cho từng skill.
- Không phá behavior cũ khi thêm skill mới.
- Dễ deploy từng policy lên embedded.

### Không chọn mega-policy ở giai đoạn đầu

Một policy duy nhất cho tất cả behavior gây:

- Observation space rất lớn.
- Reward xung đột.
- Khó phân tích nguyên nhân lỗi.
- Catastrophic forgetting khi train thêm task.
- Khó đánh giá regression.
- Khó triển khai incremental update.

### Hướng nâng cấp dài hạn

Không merge tất cả low-level policy thành một network khổng lồ.

Thay vào đó:

```text
High-Level Policy / LLM
          │
          ▼
     Skill Selector
          │
 ┌────────┼─────────┐
 ▼        ▼         ▼
Idle   Follow    Explore
 │        │         │
 └────────┼─────────┘
          ▼
     Controller
```

Đây là nền tảng để tiến tới Hierarchical RL.

---

## 4. Các policy mục tiêu

| Policy | Input chính | Output | Mục tiêu |
|---|---|---|---|
| Autonomous Idle | vision event, history, robot state | behavior primitive | hành vi tự chủ khi rảnh |
| Look At Person | face x/y | head primitive | giữ người trong vùng quan sát |
| Follow Person | person x, distance, obstacle | drive primitive | theo người ổn định |
| Play Ball | ball x, distance | drive primitive | tiếp cận/chơi bóng |
| Search Object | target history, vision, map | search primitive | tìm vật thể |
| Explore | coverage, obstacle, history | navigation primitive | khám phá vùng mới |
| Docking | dock pose, distance | drive primitive | về dock |
| Avoid Obstacle | distance/history | steering primitive | tránh vật cản |

---

## 5. Thứ tự triển khai đề xuất

```text
P0 Infrastructure
      ↓
P1 Autonomous Idle - head only
      ↓
P2 Play Ball
      ↓
P3 Follow Person
      ↓
P4 Obstacle Avoidance
      ↓
P5 Search Object
      ↓
P6 Explore
      ↓
P7 Docking
      ↓
P8 Hierarchical Skill Selection
```

Autonomous Idle được chọn làm PoC đầu tiên vì:

- Action space nhỏ.
- Không cần mô hình locomotion quá chính xác.
- Có thể bắt đầu với đầu pan/tilt.
- Reward shaping dễ kiểm chứng bằng mắt.
- Cho thấy ngay sự khác biệt giữa script và adaptive behavior.

---

## 6. Definition of Done cho nền tảng RL

Nền tảng được coi là sẵn sàng khi:

- Có thể tạo environment mới bằng config.
- Observation schema được version hóa.
- Reward component có unit test.
- Policy có metadata.
- Có deterministic evaluation.
- Có domain randomization.
- Có log TensorBoard/CSV.
- Có checkpoint.
- Có export actor-only.
- Có test inference bằng Python.
- Có đường chuyển sang C++/ESP-IDF.
- Có benchmark latency và memory.
