# Đánh giá RL của T-Rax — 14/10/2026

## Kết luận

T-Rax đã có pipeline RL hoạt động và policy behavioral thể hiện tiến bộ trong mô phỏng, nhất là vận động ở S3 và reward tương tác người ở S5. Hiện hệ thống ở giai đoạn nguyên mẫu mô phỏng, chưa có đủ bằng chứng để xác nhận sẵn sàng chạy tự chủ trên robot thật. Các điểm chặn chính là tiêu chí đánh giá quá dễ, hành vi nghỉ chưa tốt, S7 chưa đủ mượt, S6 chưa thực sự thu hút chú ý và artifact triển khai chưa đồng bộ checkpoint.

Đây là đánh giá dữ liệu cục bộ; không có telemetry phần cứng hay kết quả chạy robot thật trong phạm vi đã kiểm tra. Checkpoint behavioral mới nhất được lưu ngày 03/10/2026 lúc 16:41 theo giờ địa phương. Không suy diễn rằng robot vẫn đang huấn luyện đến 14/10.

## Phương pháp và phạm vi

- Đọc code, YAML, checkpoint, TensorBoard và manifest ONNX/int8.
- Chạy `scripts/eval.py` cho idle: 10 episode, 5 seed gốc, policy deterministic.
- Chạy benchmark có sẵn S0–S7: 3 seed, 500 bước/episode.
- Chạy `audit.py`: 3 model/baseline × 8 scenario × 3 seed × 500 bước = 36.000 bước. Seed cả `random`, `np.random` và `env.reset` để kiểm soát các nguồn RNG hiện tại. Domain randomization tắt; policy deterministic. Mỗi episode tương đương 50 giây điều khiển.
- So sánh final và checkpoint mang tên `pre_causal_fix` trên cùng môi trường hiện tại. Đây là so sánh trọng số, không tái hiện môi trường lịch sử trước sửa đổi.
- Không sửa code huấn luyện, môi trường hoặc artifact. Không huấn luyện lại, xuất lại model hay ghi firmware.

Dữ liệu chi tiết trong `metrics.json`, `training_summary.json`, mã tái chạy trong `audit.py`.

## Huấn luyện

| Policy | Bước trong checkpoint | Mạng actor | Kết quả chính |
|---|---:|---|---|
| autonomous_idle | 1.015.808 | 6 → 64 → 64 → 5, action rời rạc | Reward episode trên TensorBoard từ 64,33 lên 224,95; explained variance cuối 0,9990 |
| behavioral pre_causal_fix | 319.488 | 39 → 128 → 128 → 5, action liên tục | Learning rate trong checkpoint 0,0001 |
| behavioral final | 942.080 | 39 → 128 → 128 → 5, action liên tục | Learning rate 0,00015; chưa đạt mốc danh nghĩa 1 triệu bước |

PPO behavioral ở 10 cập nhật cuối: explained variance trung bình 0,9962; approximate KL 0,0106; clip fraction 0,1159; value loss 8,33; std action Gaussian 0,744. Critic dự đoán return tốt và cập nhật policy không cho thấy một đợt KL tăng đột biến ở đoạn này. Những số liệu đó không chứng minh policy tối ưu hoặc hành vi đúng mục tiêu.

Behavioral không có `rollout/ep_rew_mean` trong log đã đọc. Callback lấy một mẫu từ env đầu tiên mỗi đợt log, nên `reward/total` không phải reward trung bình episode. Trong event file cuối, trung bình 10 mẫu đầu là -0,2025 và 10 mẫu cuối là -0,3921; không thể dùng chúng để kết luận reward toàn bộ quá trình tăng/giảm. Tỷ lệ override ghi log giảm từ 0,388 ở mẫu đầu xuống 0,0833 ở mẫu cuối, trung bình 10 mẫu cuối 0,034; đây cũng là mẫu telemetry của một env.

## Benchmark kiểm soát RNG: final

Các giá trị dưới đây là trung bình 3 seed; cột đạt đếm số episode vượt gate hiện hành.

| Scenario | Reward/episode | Chỉ số “jerk” | Di chuyển m/episode | Đạt |
|---|---:|---:|---:|---:|
| S0 Motor sanity | -83,10 | 0,0400 | 0,0003 | 3/3 |
| S1 Khám phá góc nhìn | 35,80 | 0,0251 | 0,1441 | 3/3 |
| S2 Tò mò vật thể tĩnh | 87,11 | 0,0492 | 0,3261 | 3/3 |
| S3 Vật thể động | 152,14 | 0,0514 | 3,0868 | 3/3 |
| S4 Âm thanh | 11,96 | 0,0253 | 0,1816 | 3/3 |
| S5 Có người | 428,79 | 0,0509 | 0,9334 | 3/3 |
| S6 Thu hút chú ý | 193,96 | 0,0651 | 1,7740 | 3/3 |
| S7 Hỗn hợp | 235,80 | 0,0523 | 2,9580 | 0/3 |

S7 vượt ngưỡng 0,05 khoảng 4,6% trên trung bình; cả 3 seed đều trượt. S3/S5/S6 cũng có chỉ số trên 0,05 nhưng gate của các scenario đó không kiểm tra độ mượt. “Jerk” ở đây là trung bình tổng bình phương sai phân lệnh vận tốc giữa hai bước, không phải jerk cơ học theo m/s³ hoặc rad/s³; các kênh khác đơn vị còn được cộng chung.

Không ghi nhận bước NaN/Inf, cảnh báo MuJoCo trong episode, hoặc collision penalty ở 36.000 bước audit. Giới hạn: collision penalty là phát hiện hình học tự viết, môi trường có chặn vị trí kinematic; đây không phải chứng nhận không va chạm thực tế.

## Tiến bộ và hồi quy khi so trọng số trên cùng môi trường

- S0: reward -253,22 → -83,10; override được gắn cờ 83,27% → 0%. Tuy nhiên tỷ lệ nghỉ nghiêm ngặt giảm 13,27% → 7,33%. Policy giảm một số lỗi lệnh nhưng vẫn vận động đầu quá nhiều khi cần yên tĩnh.
- S3: reward 47,82 → 152,14; quãng đường 0,182 → 3,087 m/episode. Policy mới có vận động tích cực hơn; quãng đường chưa chứng minh tracking chính xác.
- S5: reward 241,76 → 428,79; di chuyển giảm 1,614 → 0,933 m. Phù hợp với cải thiện thưởng xã hội trong mô phỏng, chưa chứng minh chất lượng giao tiếp thật.
- S6: reward 283,01 → 193,96; độ mượt kém hơn 0,0542 → 0,0651. Không phải mọi năng lực đều cải thiện.
- S7: reward 157,50 → 235,80, nhưng độ mượt 0,0259 → 0,0523 và chuyển từ đạt sang trượt. Có đánh đổi giữa hoạt động và độ mượt.

Idle chạy lại: reward 233,82 ± 170,60, tracking success 99,9%, repetition 60,0%, REST 40,0%; gate có sẵn báo đạt. Độ phân tán reward lớn và repetition sát ngưỡng 60%; đây là kết quả tracking đơn giản của môi trường idle, không tương đương chất lượng behavioral hay robot thật.

## Vì sao chưa thể tin hoàn toàn vào gate hiện tại

1. **Baseline cũng đạt 8/8.** Action cố định `[0, 0, 0, 0, 0.1]`, qua cùng lớp phản xạ, vượt mọi gate trên cả ba seed. Đây là bằng chứng trực tiếp rằng gate chưa phân biệt được policy học tốt với baseline rất đơn giản. Baseline tốt hơn final ở reward S0/S1/S2/S4, final tốt hơn ở S3/S5/S6/S7; reward đơn lẻ cũng không đủ phản ánh hoàn thành nhiệm vụ.
2. **Nghỉ bị chấm quá dễ.** `stationary_ratio` chỉ yêu cầu tổng trị tuyệt đối 4 lệnh < 0,50. Final S0 đạt 99% theo chỉ số này, nhưng chỉ 7,33% theo điều kiện nghỉ của internal drives: tổng < 0,08 và attention < 0,2. Reward S0 âm, với penalty excessive movement chiếm ưu thế ở episode kiểm tra.
3. **S6 có success giả định 85% khi không thử thu hút chú ý.** Cả baseline, pre và final đều có mean escalation tier 0, không có attention attempt. `compute_metrics()` trả mặc định 0,85 nếu không có attempt, nên gate ≥70% vẫn đạt. Spam 0% ở đây chưa chứng minh robot biết phát tín hiệu đúng lúc.
4. **Inspection không yêu cầu thực sự nhìn đúng vật thể.** Bộ đếm object inspection tăng khi target_visible liên tục >0,3 giây; `target_visible` truyền vào metrics chỉ dựa trên khoảng cách/có người, khác với visibility có FOV trong observation. Một lần inspection không xác nhận tiếp cận và khảo sát vật thể.
5. **Handoff chưa đo độ chính xác tracking.** Bộ đếm tự kích hoạt khi target_visible và hết cooldown; baseline S3/S7 cũng có 10 handoff/episode. Không có sai số tracking hay thực thi chuyên gia tracking được xác nhận bởi chỉ số này; S7 còn có thể gán request dynamic_ball cho target người đang được ưu tiên.
6. **Override flag không đếm mọi lần thay đổi action.** Final báo 0% override ở audit, nhưng action sau xử lý khác action policy ở 88–100% bước tùy scenario. Khác biệt bao gồm clipping, deadband, smoothing, giữ đế IDLE và phản xạ; không nên gọi tất cả là can thiệp khẩn cấp. Tuy nhiên 0% flag không có nghĩa policy được thực thi nguyên vẹn.
7. **Bộ evaluation cũ có điều kiện tổng thiếu gate.** `evaluate_behavioral.py` tính head scan và approach nhưng không đưa chúng vào `overall_pass`; cách cấu hình “empty_room” sau reset vẫn giữ ScenarioManager mặc định S7. Tỷ lệ safety override trong script này lấy max của bộ đếm reset mỗi episode rồi chia tổng bước, gây thiếu đếm. Do đó không sử dụng tuyên bố READY của script này làm kết luận.

## Mô phỏng và triển khai

- Đế robot được điều khiển bằng cập nhật trực tiếp `qpos` X/Y/yaw, có clamp biên và đẩy ra khỏi vật cản. Scene chỉ có actuator pan/tilt, chưa có động lực học motor/xích hoàn chỉnh. Quan sát vận tốc sử dụng lệnh action, còn embedding/nhận dạng đối tượng được tổng hợp từ trạng thái mô phỏng. Bằng chứng hiện tại phù hợp với học hành vi cấp cao hơn là kiểm chứng điều khiển xích và perception thực.
- Domain randomization trong reset chỉ áp dụng S7 và chủ yếu là motor gain, noise ToF, vị trí/sự hiện diện người; audit chưa kiểm tra randomization hay episode dài hơn 50 giây. Có `servo_delay_steps` nhưng chưa thấy cơ chế trễ servo được thực thi.
- `MUJOCO_LOG.TXT` có 6 cảnh báo NaN/Inf/huge QPOS/QVEL lúc 03/10/2026 16:49, sau thời gian lưu final. Audit hiện tại không tái hiện cảnh báo; không quy kết rằng checkpoint đã bị huấn luyện bằng dữ liệu hỏng. Cần tái hiện điều kiện viewer/control đã gây lỗi. Env hiện không terminate khi simulation lỗi.
- ONNX behavioral được lưu 03/10 lúc 06:33, trước final lúc 16:41. So trực tiếp initializer ONNX với actor final cho thấy trọng số khác nhau, ví dụ max absolute difference của first-layer weights 0,8062 và action-layer weights 0,4991. Manifest không gắn SHA của checkpoint. Chưa thể dùng artifact này để đại diện policy final.
- Có header int8 và manifest ESP32-S3, nhưng trong phạm vi này chưa có đo sai lệch hành vi FP32/ONNX/int8, latency thực trên ESP32-S3 hay telemetry triển khai thật. `verify_pipeline.py` đặt tiêu đề kiểm tra int8 nhưng phần consistency chỉ tính PyTorch logits, không chạy int8 để so.
- Config YAML reward không được truyền vào `BehavioralRewardMixer()` trong pipeline behavioral; hiện default trùng nhiều giá trị YAML nhưng chỉnh YAML sẽ không tự đổi reward. Tương tự một số field environment không được truyền từ config vào env.
- Khi resume, script tính `actual_total_timesteps = model.num_timesteps + requested`, rồi gọi SB3 với `reset_num_timesteps=False`. SB3 lại cộng số bước hiện tại trong `_setup_learn`; do đó số bước chạy thêm có thể lớn hơn số bước người dùng yêu cầu. Cần sửa để ngân sách huấn luyện và tiến độ báo cáo chính xác.

## Thứ tự công việc tiếp theo

1. Sửa bộ đánh giá trước: S6 yêu cầu có attempt thật, tính success trên sự kiện; inspection yêu cầu gaze/distance/dwell đúng; tracking đo sai số và thời gian mất dấu; ghi tỷ lệ sửa action theo từng nguyên nhân; gate tổng bao gồm mọi tiêu chí và áp dụng độ mượt nhất quán.
2. Bổ sung đánh giá S0 theo nghỉ nghiêm ngặt, giảm vận động đầu không có kích thích; xử lý độ mượt S7 mà vẫn giữ phản xạ thoát nguy hiểm nhanh. Kiểm tra từng seed thay vì chỉ trung bình.
3. Sửa truyền config và budget resume; dùng Monitor/EvalCallback để có episode return/success, lưu best checkpoint theo validation độc lập. Không chọn final chỉ dựa vào số bước.
4. Mở rộng validation ngoài 3 seed, bật domain randomization, episode dài và vị trí chưa gặp; so với baseline và tách đóng góp của phản xạ an toàn. Không loại bỏ lớp an toàn để thử trên phần cứng.
5. Sau khi đạt tiêu chí, export đúng checkpoint với checksum, kiểm tra parity FP32 → ONNX → int8 và đo latency 10 Hz trên phần cứng. Xác minh mô hình motor/perception trước thử tự chủ thực tế.

Ưu tiên hiện tại là làm cho số đo phản ánh đúng năng lực rồi mới tăng ngân sách huấn luyện. Huấn luyện thêm với gate hiện tại có thể cải thiện reward nhưng vẫn bỏ sót kỹ năng cần thiết.
