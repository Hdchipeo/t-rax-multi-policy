# Kết quả triển khai purposeful v2 — 14/10/2026

Đã triển khai mục tiêu có vòng đời, bộ nhớ theo episode, kỹ năng có điều kiện kết thúc và learned residual; sửa internal drives, RNG, event retrigger, điều khiển servo, gate và pipeline huấn luyện/validation. Viewer và exporter hỗ trợ contract 54D. Chế độ v1 vẫn dùng observation 39D nhưng cần đánh giá lại với môi trường đã sửa.

## Kiểm chứng

- 17 test unittest: đạt; gồm timestep invariance, pin/readiness, goal completion/memory, safety interruption, motion wakeup, attribution phản hồi, gate không có attempt, inspection, tracking, hard constraints sau slew, sự kiện một lần và replay cùng seed.
- Bộ verify behavioral hiện có: 7 nhóm kiểm tra đạt; Gym checker đạt cho cả rl và goal_residual; Python compilation đạt.
- Huấn luyện smoke: 2.048 bước mới + resume thêm 1.024 + 1.024 = 4.096 bước. Xác nhận resume không cộng ngân sách hai lần; validation và checkpoint hoạt động. Chưa phải training tới hội tụ.
- Nominal goal-skill controller: 24/24 episode đạt, seed 1001–1003, 500 bước/episode.
- Checkpoint 4.096 bước: 80/80 episode đạt, seed mới 2001–2010, 1.000 bước/episode (100 giây), bật domain randomization. Tổng 80.000 bước. Không episode collision penalty hoặc simulation instability. Domain randomization hiện chỉ thay nhiều tham số ở S7; không coi S0–S6 là thử nghiệm nhiều bố trí vật thể khác nhau.
- FP32/ONNX: graph hợp lệ, 32 input kiểm tra numerical parity; max absolute error 2.98023224e-08. Chưa kiểm tra int8 hoặc ESP32-S3 latency.
- Quantizer từ chối model 54D thay vì phát manifest v1 sai contract.
- Baseline đứng yên v1 trượt inspection/tracking/attention theo gate mới. Checkpoint behavioral v1 trượt nghỉ nghiêm ngặt S0; gate không còn báo thành công chỉ vì model tồn tại.

## Kết quả mở rộng

Các giá trị là trung bình trên 10 episode từng scenario. Cột tracking chỉ có ý nghĩa khi có moving-target steps; cột tín hiệu chỉ có ý nghĩa khi có người.

| Scenario | Episode đạt | Nghỉ nghiêm ngặt | Command variation | Attention attempt/episode | Tracking aligned |
|---|---:|---:|---:|---:|---:|
| S0 | 10/10 | 100.0% | 0.00000 | 0.0 | 0.0% |
| S1 | 10/10 | 82.0% | 0.00027 | 0.0 | 0.0% |
| S2 | 10/10 | 75.9% | 0.00089 | 0.0 | 0.0% |
| S3 | 10/10 | 62.9% | 0.00316 | 0.0 | 89.6% |
| S4 | 10/10 | 80.1% | 0.00213 | 0.0 | 0.0% |
| S5 | 10/10 | 79.0% | 0.00103 | 3.0 | 0.0% |
| S6 | 10/10 | 77.1% | 0.00119 | 3.0 | 0.0% |
| S7 | 10/10 | 43.7% | 0.00749 | 1.8 | 63.1% |

JSON gốc: `nominal_validation.json`, `extended_validation.json`; có target ID/phase transitions. Không so trực tiếp reward với báo cáo ngày 14/10 trước sửa vì động lực học, drives, reward và gate đã thay đổi.

## Phạm vi và giới hạn

Kết quả hiện chủ yếu đến từ cấu trúc mục tiêu và kỹ năng nền tảng được viết bằng luật, cộng một residual nhỏ. Chưa có bằng chứng residual 4.096 bước tốt hơn nominal controller; nominal cũng vượt gate. Đánh giá video bởi con người vẫn cần để xác nhận cảm giác tự nhiên.

Bộ nhớ chỉ tồn tại trong episode. Đế kinematic, perception tổng hợp, mô hình phản ứng người còn đơn giản; S7 chưa bắt buộc đủ mọi kỹ năng con trong mỗi episode. Không phải chứng nhận chạy robot thật. Gate và seed dùng để phát triển vẫn cần một tập kiểm thử cuối độc lập nếu tiếp tục tuning.

Firmware v2/int8 chưa triển khai trong repo này; không có thao tác nạp robot. Bản ONNX trong `export_smoke` có `deployment_ready=false`. Các checkpoint và artifact v1 được giữ lại.

## Sử dụng

Xem README ở gốc dự án để chạy viewer, đánh giá hoặc huấn luyện đầy đủ. Lần training tiếp theo dùng config v2 mặc định; resume model 54D, không resume trực tiếp model 39D.
