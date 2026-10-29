# Huấn luyện RL cho bối cảnh bàn làm việc có người

Policy mới dùng PPO `MultiInputPolicy`, điều khiển trực tiếp năm kênh của môi trường bàn. Bốn kênh đầu điều khiển tiến/lùi, quay thân và pan/tilt. Kênh thứ năm vẫn là placeholder biểu cảm, chưa có đầu ra LED/âm thanh. Không sử dụng motor command từ `DeskDemo`, controller mẫu hay checkpoint phòng.

## Chạy lại

```sh
.venv/bin/python scripts/train_desk.py --compare-baselines
# Kiểm tra kết nối nhanh, không dùng để kết luận năng lực:
.venv/bin/python scripts/train_desk.py --timesteps 2048 --n-envs 1
# Tiếp tục checkpoint; mỗi lệnh ghi ra thư mục run mới:
.venv/bin/python scripts/train_desk.py --resume checkpoints/TEN_RUN/final_model.zip --timesteps 262144 --lr 0.0001 --entropy 0.005
# Chạy policy đã học trong viewer:
.venv/bin/python scripts/run_desk.py --policy checkpoints/TEN_RUN/best_model.zip
# Đánh giá độc lập:
.venv/bin/python scripts/train_desk.py --evaluate checkpoints/TEN_RUN/best_model.zip --output reports/desk_social_eval
```

Tên run hiện tại nằm trong `checkpoints/desk_social_current_run.txt`. Trong mỗi run:

- `config.yaml`: cấu hình đầy đủ của lần chạy.
- `train.log`, `tensorboard/`, `status.json`: tiến độ và telemetry.
- `latest_model.zip`, `best_model.zip`, `final_model.zip`: checkpoint; best được chọn trên validation, không dựa test.
- `*.metadata.json`: SHA checkpoint, mã nguồn/scene/mesh, phiên bản thư viện, observation, seed và thông số PPO thực tế.
- `baselines.json`: robot đứng yên, action ngẫu nhiên, mạng chưa train.
- `best_validation.json`, `test_evaluation.json`: kết quả seed riêng.

Checkpoint cũ được giữ nguyên. Khởi tạo mới dùng seed 42, CPU một thread cho mạng và bốn subprocess môi trường. Đợt đầu có 262.144 transitions, rollout 512 bước/môi trường, batch 256, 6 epochs, MLP 128–128, learning rate 3e-4, gamma .995 và entropy .01. Safety và vật lý xích luôn bật.

## Curriculum và contract

Mỗi episode train dài 600 bước (30 giây). 35% reset bắt đầu ở đầu lịch; các reset còn lại lấy mốc quanh lúc có người, vẫy tay, đặt vật, đứng lên hoặc bàn vắng người. Reset replay các mốc thả vật để các episode bắt đầu muộn vẫn có ly/đồ chơi. Pose robot ban đầu, pan/tilt và góc thân ngẫu nhiên trong giới hạn vùng; thời gian người xuất hiện dịch 0–2 giây theo seed. Hiện chưa thay đổi bố trí bàn, ngoại hình người hoặc độc lập từng mốc sự kiện.

Contract `desk_social_rl_v1` có 71 giá trị dạng Dict: 61 giá trị của `desk_social_dict_v3` cộng `memory` (5) và `elevation` (5). Memory gồm ba cờ đã nhìn nhận người và hai cờ đã quan sát vật hoàn tất; chỉ cập nhật khi có dữ liệu trong tầm nhìn. Elevation là sai số nhìn dọc/π của người/vật nhìn thấy; ngoài FOV hoặc bị khuất thì bằng 0. Không đưa lịch tương lai, thời gian episode hoặc hướng mục tiêu bị khuất vào actor.

Các vị trí simulator được dùng cho phần thưởng định hướng và bảo vệ, có ghi `privileged_reward_shaping=True`. Perception là detector giả lập có FOV/occlusion, chưa phải neural vision chạy trên ảnh thật.

Để tăng tốc, animation mắt xích chỉ cập nhật ở cuối bước điều khiển khi train. Tiếp xúc `geom_surfacevel` và tích phân vật lý 500Hz vẫn giữ nguyên; test so sánh cho qpos/qvel giống nhau trong dung sai 1e-10.

## Nhiệm vụ và đánh giá

- **Nhìn nhận người:** nhìn thấy, hướng camera trong sai số góc tổng hợp <0,28 rad, thân gần đứng yên, duy trì ít nhất 0,4 giây. Mỗi ID được thưởng hoàn tất một lần/episode.
- **Đáp lại vẫy tay:** có sự kiện vẫy thực sự, đạt điều kiện nhìn 0,4 giây trong cửa sổ 3 giây. Không tính phản ứng trước sự kiện; mỗi sự kiện chỉ thưởng một lần. “Đáp lại” ở đây là hướng nhìn, chưa phải phát lời nói/cử chỉ đáp lễ.
- **Quan sát vật mới:** vật đã được thả, nhìn thẳng và thân gần đứng yên liên tục ≥0,6 giây, cự ly <0,8m, vật gần tĩnh. Mỗi vật chỉ thưởng một lần. Dwell có thể bao gồm thời gian nhìn vật lúc đang được cầm; chỉ công nhận hoàn tất khi vật đã thả, ở trong cự ly và gần tĩnh.
- Thưởng nhỏ khi lần đầu thấy người/vật; shaping theo chênh lệch sai số nhìn, không cộng điểm liên tục vì giữ nguyên hướng nhìn.
- Phạt đổi lệnh gấp, năng lượng, safety override, va chạm, rơi và bất ổn. Khi không còn nhiệm vụ, giảm động cơ chạy vô ích.

Validation chạy mỗi 65.536 bước trên seed 1001–1002, episode đầy đủ 90 giây; test cuối dùng seed 2001–2003 với yaw đầu 0, π, -π/2. Cả baseline và policy dùng cùng bộ test. Thay đổi seed hiện chủ yếu thay độ trễ lịch và vị trí bóng, không chứng minh tổng quát sang bàn hay người khác.

Gate thử nghiệm: không va chạm/rơi/bất ổn; tỷ lệ đáp nhìn vẫy tay ≥50%; trung bình nhận biết ≥2 người, quan sát ≥1 vật; safety override ≤5%. `best_model` ưu tiên không có episode mất an toàn, sau đó số nhiệm vụ đạt, mức can thiệp và reward. Tên best chỉ có nghĩa tốt nhất trong đợt train; chỉ `passed` trong báo cáo mới xác nhận qua đủ gate. Gate chưa đạt thì checkpoint được giữ làm kết quả thử nghiệm, chưa dùng thay model chạy thật.

## Đợt thực nghiệm đầu

Đã chạy 262.144 bước khởi tạo mới và một lượt fine-tune thêm 262.144 bước (tổng 524.288). Fine-tune giữ reward/observation/vật lý, giảm learning rate xuống 1e-4 và entropy xuống .005. Kết quả, checkpoint được chọn và các gate chưa đạt được ghi trong `reports/desk_social_rl/training_report.md`; không suy ra đạt năng lực chỉ từ việc train hoàn tất.
