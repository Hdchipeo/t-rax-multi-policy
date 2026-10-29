# T-Rax purposeful behavior v2

Chế độ mới tổ chức hành vi thành mục tiêu có vòng đời: định hướng → tiếp cận → quan sát/phát tín hiệu → chờ phản hồi → hoàn thành hoặc bỏ cuộc → nghỉ. Mục tiêu giữ nguyên target ID trong khi thực hiện. Bộ nhớ theo episode giảm việc kiểm tra lại mục tiêu không thay đổi; chuyển động mới có thể đánh thức robot đang nghỉ. Phản xạ an toàn có quyền ngắt mục tiêu.

## Kiến trúc

- `GoalManager`: chọn mục tiêu REST, INSPECT, TRACK, SOCIAL, EXPLORE, ORIENT; deadline, tiến triển, cooldown, bộ nhớ vật thể/góc nhìn.
- `GoalSkillPolicy`: kỹ năng điều khiển có nhịp nhìn/chờ/dừng; PPO học hiệu chỉnh nhỏ trên kỹ năng. Nghiêng đầu và nhìn đi chỗ khác có đích ổn định, không đổi hướng ngẫu nhiên mỗi bước.
- `BehaviorManager`: xử lý an toàn, giới hạn vận tốc/gia tốc, chặn lùi mù, giới hạn servo; ghi riêng các nguyên nhân sửa lệnh. Ràng buộc cứng được áp lại sau smoothing.
- `InternalDrivesEngine`: cập nhật theo giây; pin không tự tăng khi nghỉ, readiness có thể phục hồi; nhu cầu xã hội giảm khi có engagement.
- `evaluation`: đánh giá từng episode trên seed độc lập. Một seed trượt khiến scenario chưa đạt; reward cao không thay thế tiêu chí kỹ năng.

Các hành vi nền tảng hiện được viết bằng luật. Checkpoint 4.096 bước là kiểm tra pipeline/residual, chưa chứng minh RL đã tự học toàn bộ kỹ năng hay được tối ưu đến hội tụ.

## Chạy mô phỏng

Từ thư mục dự án, dùng môi trường `.venv` hiện có.

```sh
# Xem checkpoint v2; trên macOS dùng mjpython cho native viewer.
.venv/bin/mjpython scripts/visualize_policy.py --scenario S7 --log-style detailed

# Xem kỹ năng nền tảng, không có learned residual.
.venv/bin/mjpython scripts/visualize_policy.py --env purposeful --nominal --scenario S2

# Đánh giá checkpoint, bật domain randomization.
.venv/bin/python scripts/evaluate_purposeful.py --model checkpoints/purposeful_behavioral_v2/final_model.zip --randomize

# Chạy tests.
.venv/bin/python -m unittest discover -s tests -v
```

Viewer hiển thị goal type, target ID, phase và lệnh thực thi. Model không tồn tại hoặc sai schema sẽ báo lỗi; baseline chỉ chạy khi chọn `--nominal`.

## Huấn luyện

```sh
# Mặc định dùng configs/policies/purposeful_v2.yaml, khởi tạo mới.
.venv/bin/python scripts/train_behavioral.py

# Resume: --timesteps là số bước muốn chạy THÊM.
.venv/bin/python scripts/train_behavioral.py --pretrained checkpoints/purposeful_behavioral_v2/final_model.zip --timesteps 100000 --n-envs 8
```

PPO có thể làm tròn ngân sách lên theo kích thước rollout. Script không cộng timestep hiện tại hai lần. Monitor ghi episode return/length; telemetry tổng hợp các env; validation định kỳ chọn `best_model.zip` theo tỷ lệ đạt gate trước, reward sau. `final_model.zip` là model cuối lượt chạy. Sidecar metadata ghi SHA checkpoint, SHA nguồn runtime và config. Khi thay runtime/config, đánh giá best cũ không được dùng trực tiếp để xếp hạng lượt mới.

Checkpoint thử hiện có: `checkpoints/purposeful_behavioral_v2/final_model.zip`, 4.096 bước. Không dùng số bước nhỏ này như bằng chứng policy đã hội tụ.

## Contract v2 và tương thích

Observation `purposeful_obs_v2` có 54 chiều:

| Index | Nội dung |
|---|---|
| 0–38 | Behavioral observation hiện có |
| 39–44 | One-hot goal: REST, INSPECT, TRACK, SOCIAL, EXPLORE, ORIENT |
| 45–48 | One-hot phase: ORIENT, APPROACH, OBSERVE, WAIT |
| 49 | Thời gian mục tiêu / deadline, giới hạn [0,1] |
| 50 | Tiến triển mục tiêu |
| 51–53 | Fear, arousal, readiness |

Action PPO có 5 chiều trong [-1,1]: signed residual. Hiệu chỉnh motor/head có scale `[0.04, 0.08, 0.08, 0.06]`; attention được hiệu chỉnh tối đa 0,08 trong giai đoạn phát tín hiệu xã hội. REST bỏ residual để giữ yên. Các phase quan sát/chờ không được tiến về trước. Nominal action sau phối hợp residual vẫn đi qua safety.

Mạng 39D cũ vẫn có chế độ `controller_mode: rl`, dùng `configs/policies/behavioral_dynamics_v1.yaml`. Không thể resume trực tiếp trọng số 39D vào mạng 54D. Các sửa lỗi drives, sự kiện, servo và gate làm thay đổi môi trường/đánh giá v1; checkpoint cũ phải được đánh giá lại, không kế thừa kết quả pass trước đây.

## Đánh giá

- S0 dùng nghỉ nghiêm ngặt: tổng trị tuyệt đối 4 lệnh <0,08 và attention <0,2; cần ≥50% thời gian.
- Inspection cần target trong FOV, sai số nhìn <0,2 rad, cự ly 0,35–0,8 m, base gần đứng yên, mục tiêu không chuyển động, dwell ≥1 giây. Cùng target chỉ đếm một lần trong episode.
- Tracking chỉ tạo handoff cho vật thể đang chuyển động; S3 cần visual alignment ≥60%, có timeout/lost-target metrics.
- Attention đếm cạnh bắt đầu tín hiệu tier≥2, có cooldown 3 giây và cửa sổ phản hồi 2 giây. Không có attempt ⇒ success 0, gate S6 trượt.
- Mọi scenario kiểm tra command variation ≤0,05, servo saturation ≤15%, không collision penalty và không simulation instability.
- NaN/Inf/mất ổn định vật lý kết thúc episode với penalty; action không hữu hạn bị từ chối.

“Motion jerk” trong log là command variation, chưa phải jerk cơ học theo SI. Collision vẫn là phép kiểm tra hình học mô phỏng. S7 hiện kiểm tra tích hợp/an toàn/độ mượt; không yêu cầu hoàn thành mọi kỹ năng con trong mỗi episode.

Kết quả và giới hạn xem `reports/purposeful_v2/implementation.md`; JSON chứa episode metrics và goal transitions để kiểm tra vì sao robot chọn/dừng nhiệm vụ.

## Export và robot thật

Exporter tự nhận observation dimension, kiểm tra FP32/ONNX và ghi SHA nguồn checkpoint, output semantics và action bounds. ONNX output là actor mean; runtime phải clip vào action bounds rồi thực hiện goal/skill/safety. Bản kiểm tra nằm trong `reports/purposeful_v2/export_smoke/`, chưa phải artifact firmware phát hành.

Quantizer hiện chặn v2 để tránh tạo nhầm manifest v1: firmware hiện tại chưa được port goal manager, bộ nhớ và residual contract. Cần runtime tương ứng, dữ liệu calibration thực và kiểm tra int8/latency trước triển khai ESP32-S3. Artifact v1 hiện có không được thay bằng model thử v2.

Perception hiện tổng hợp từ trạng thái MuJoCo; mobile base còn kinematic; domain randomization hiện chủ yếu áp dụng S7. Chưa có kiểm chứng motor/xích, cảm biến camera/ToF thật, đánh giá video của người dùng hay thử trên robot thật.


## Môi trường bàn làm việc

Đã thêm scene bàn 1,40 × 0,75 m và vùng riêng T-Rax 0,50 × 0,40 m.
Chạy `.venv/bin/python scripts/run_desk.py` để mở demo.
Xem [hướng dẫn desktop environment](docs/desk_environment.md) và ảnh trong `reports/desk_environment/`.
Desk dùng contract RL riêng; không nạp trực tiếp checkpoint phòng cũ.

Scene hiện có tối đa ba người, chuyển động ngồi/đứng/đi lại, cử chỉ và sự kiện đặt ly nước/đồ chơi. Xem [kịch bản tương tác con người](docs/desk_human_scenarios.md); quan sát RL mới là `desk_social_dict_v3`.

Đã có PPO cho bối cảnh bàn có người và vật mới: [hướng dẫn train/chạy policy](docs/desk_social_rl.md), [báo cáo huấn luyện](reports/desk_social_rl/training_report.md). Chạy viewer với `--policy <checkpoint.zip>` để dùng mạng đã học.
