# Cơ sở lý thuyết RL áp dụng cho T-RAX

## 1. Reinforcement Learning

Một bài toán RL có thể mô tả bằng:

```text
State/Observation
      ↓
    Policy
      ↓
    Action
      ↓
 Environment
      ↓
Reward + next observation
      └──────────────↺
```

Policy ký hiệu:

\[
a_t = \pi_\theta(o_t)
\]

Trong đó:

- \(o_t\): observation tại thời điểm t.
- \(a_t\): action.
- \(\pi\): policy.
- \(\theta\): trọng số neural network.

Mục tiêu training là tìm \(\theta\) để cực đại tổng reward dài hạn:

\[
J(\theta)=E[\sum_{t=0}^{T}\gamma^t r_t]
\]

---

## 2. Observation và State

Robot thật không biết toàn bộ state của thế giới. Camera, ToF, IMU chỉ cung cấp observation.

Ví dụ Autonomous Idle:

```text
person_visible
person_x
motion_level
obstacle_distance
head_pan
head_tilt
time_since_person_seen
previous_action
```

Không nên đưa raw RGB vào policy V1.

Vision model nên chuyển ảnh thành semantic state trước:

```text
OV2640
  ↓
YOLO
  ↓
Semantic Observation
  ↓
RL Policy
```

Điều này giảm rất mạnh kích thước policy và Sim-to-Real gap.

---

## 3. Action Space

Có hai kiểu chính.

### Continuous

```text
left_motor  ∈ [-1, 1]
right_motor ∈ [-1, 1]
```

Phù hợp low-level motion.

### Discrete

```text
REST
LOOK_LEFT
LOOK_RIGHT
TRACK_PERSON
TURN_LEFT
TURN_RIGHT
MOVE_FORWARD
```

Phù hợp Autonomous Idle.

Đề xuất T-RAX:

- Idle: discrete behavior primitive.
- Navigation skill: continuous hoặc hybrid.
- Safety: deterministic, không RL.

---

## 4. Reward Shaping

Reward không mô tả cách robot phải làm; reward mô tả điều gì được xem là tốt/xấu.

Ví dụ Idle:

\[
R =
w_aR_{attention}
+w_nR_{novelty}
+w_sR_{smooth}
-w_cR_{collision}
-w_rR_{repetition}
-w_eR_{energy}
\]

Sai reward có thể tạo reward hacking.

Ví dụ chỉ reward `TRACK_PERSON` khi thấy người có thể khiến robot stare liên tục.

Do đó cần:

- diminishing reward theo thời gian;
- penalty cho lặp;
- energy cost;
- rest reward trong môi trường không có sự kiện.

---

## 5. PPO

PPO — Proximal Policy Optimization — phù hợp làm baseline vì:

- dễ dùng;
- ổn định tương đối;
- hỗ trợ action continuous/discrete;
- có implementation tốt trong Stable-Baselines3.

Một policy nhỏ:

```text
Observation 16
      ↓
Dense 64 + ReLU
      ↓
Dense 64 + ReLU
      ↓
Action logits / action mean
```

Trong training PPO thường có:

```text
Actor  -> chọn action
Critic -> estimate value
```

Khi deploy robot thật thường chỉ cần Actor.

---

## 6. Partial Observability và Memory

Idle behavior phụ thuộc history.

Ví dụ:

```text
person_visible = false
```

không cho biết người chưa từng xuất hiện hay vừa ra khỏi khung hình.

Có thể giải quyết theo ba cấp:

### V1 — explicit history

```text
last_person_x
time_since_person_seen
previous_action
```

### V2 — frame stacking

```text
obs[t]
obs[t-1]
obs[t-2]
```

### V3 — recurrent policy

```text
MLP → GRU/LSTM → Action
```

Khuyến nghị bắt đầu V1.

---

## 7. Curriculum Learning

Không train full task ngay.

Ví dụ Autonomous Idle:

```text
Level 1: REST + LOOK
Level 2: person appear/disappear
Level 3: tracking
Level 4: body rotation
Level 5: locomotion
Level 6: obstacles
```

Mỗi level chỉ tăng một nhóm độ khó.

---

## 8. Domain Randomization

Simulation không giống robot thật hoàn toàn.

Mỗi episode có thể random:

```text
motor_gain
servo_gain
motor_delay
camera FOV
sensor noise
person speed
object position
floor friction
battery response
```

Mục tiêu không phải mô phỏng một robot "hoàn hảo", mà train policy chịu được một phân phối robot/môi trường.

---

## 9. Sim-to-Real

Pipeline:

```text
MuJoCo
 ↓
Gymnasium
 ↓
PPO training
 ↓
Evaluation
 ↓
Domain randomization
 ↓
Actor export
 ↓
Quantize
 ↓
ESP32 inference
 ↓
Shadow mode
 ↓
Limited real control
 ↓
Full deployment
```

Không nên nhảy trực tiếp từ policy simulation sang full motor control.

---

## 10. Hierarchical RL

Khi nhiều skill đã ổn định:

```text
High-Level Policy
      ↓
Skill ID
      ↓
Low-Level Policy
      ↓
Action
```

High-level policy có thể học:

```text
IDLE
FOLLOW
PLAY_BALL
SEARCH
EXPLORE
```

Low-level policy vẫn tách biệt.

Đây là hướng mở rộng phù hợp hơn all-in-one network.
