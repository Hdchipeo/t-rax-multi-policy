# Đánh giá project và chuyển policy mới sang T-Rax thật

Ngày kiểm tra: 15/10/2026. Phạm vi: repo RL `t-rax-multi-policy`, board T-Rax trong `t-rax-main`, và firmware thị giác trong `t-rax-vision-det-pico/04_deploy_firmware`. Đây là đánh giá mã nguồn và checkpoint, không phải xác nhận firmware đang chạy trên thiết bị.

Người dùng xác nhận robot **chưa có encoder bánh xích, IMU và cảm biến nhìn xuống chống rơi**.

## Kết luận triển khai

Project đã đạt mức prototype RL có thể tái lập trong một scene MuJoCo. Policy mới đã học được hành vi có ích, nhưng chuỗi perception → observation → inference → actuator → safety trên robot chưa tương thích. Có thể bắt đầu công việc chuyển giao và kiểm tra không cấp quyền motor; chưa đủ cơ sở cho xích tự do chạy trên bàn.

Hướng ưu tiên: chuẩn hóa dữ liệu thật, kiểm chứng export, sau đó thử điều khiển đầu với thân cố định và quyền điều khiển motor bị khóa độc lập. Policy dành cho cấu hình thân cố định phải được đánh giá/huấn luyện đúng cấu hình đó. Sau khi giải quyết đo chuyển động và chống rơi mới mở điều khiển xích.

## 1. Bằng chứng hiện có

- Tổng train: 524.288 transitions PPO, khởi tạo mới và fine-tune. Checkpoint best tại bước 458.752, không phải checkpoint cuối.
- Checkpoint: `checkpoints/desk_social_ppo_v1_finetune_20261015_135543/best_model.zip`.
- Actor thực tế: `71 → 128 → Tanh → 128 → Tanh → 5`, 26.373 tham số. Khoảng 103 KiB nếu lưu toàn bộ tham số FP32; con số này chưa tính runtime, activation, audio, vision hoặc mạng. Đây là mạng tương đối nhỏ, nhưng chưa có benchmark chính model này trên S3.
- Test 3 seed: đáp nhìn vẫy tay 50%, nhìn nhận 2/3 người, quan sát 1/2 vật.
- Audit 5 seed/góc thân khác: đáp nhìn 65%, nhìn nhận 2/3 người, quan sát 1/2 vật.
- 8 episode test/audit, mỗi episode 90 giây: không va chạm/rơi/bất ổn, safety override bằng 0. Các episode vẫn có safety bật; không suy ra policy tự an toàn khi bỏ safety.
- Đã chạy lại toàn bộ **46 tests: đạt**, 17 giây. Tests Python không thay thế build firmware hoặc test phần cứng.
- Đã gọi exporter hiện tại với chính checkpoint mới: dừng ở `TypeError: 'NoneType' object is not subscriptable` vì dùng `model.observation_space.shape[0]` trên Dict.
- Không sửa firmware, nạp model hay gửi lệnh đến robot trong lần đánh giá này.

Các seed chủ yếu đổi độ trễ chung của lịch và bóng; không phải tám môi trường độc lập. Lịch tương tác, định danh người, bố trí bàn và vị trí đặt ly/đồ chơi vẫn có cấu trúc cố định. “Đáp lại” hiện là hướng nhìn có dwell, không phải nhận biết ý định hoặc giao tiếp hai chiều. Còn bỏ lỡ một người và một vật; chưa có đánh giá mức tự nhiên bằng người quan sát, endurance, latency sensor thật hoặc tình huống lỗi.

## 2. Những điểm chặn đã xác minh

| Ưu tiên | Điểm chặn | Bằng chứng / tác động |
|---|---|---|
| P0 | Chưa chống rơi độc lập | Robot chưa có cliff sensors; geofence mô phỏng dùng pose thật của simulator. ToF phía trước không xác nhận mặt bàn còn dưới xích. |
| P0 | ToF chưa được khởi tạo trong source board | `t_rax_board.cc:155,161`: `Initialize()` và `StartTask()` đang comment. Chưa xác nhận trạng thái trên firmware đã flash. `SafetySupervisor` chỉ kiểm tra khoảng cách dương; dữ liệu -1 không tạo chặn vật cản trong nhánh này. |
| P0 | Actor cần dữ liệu robot chưa đo được | Body velocity, yaw rate, roll/pitch, pose trong vùng và khoảng cách biên hiện lấy từ MuJoCo. Chưa thấy nguồn encoder/IMU/localization trong WorldState, phù hợp xác nhận phần cứng của người dùng. |
| P0 | Chưa có pipeline đa mục tiêu tương ứng | WorldState giữ một target. Firmware vision có nhiều detection cho debug nhưng đường gửi sang robot chọn `results.front()`. Không có stable IDs, nhãn vẫy tay/ngồi/đi lại, khoảng cách metric tới từng người hay trạng thái vật đang cầm như contract mới. |
| P0 | Runtime từ chối kích thước mới | `policy_runtime.h:37`: `MAX_INPUT_DIM=48`, trong khi actor cần 71. Adapter đang tạo `float obs[39]`. |
| P0 | Sai activation nếu dùng quantizer cũ | Actor mới có hai Tanh; runtime chỉ có cấu hình `use_relu`, còn quantizer giả định hidden ReLU. Không được đổi Tanh sang ReLU chỉ bằng thay header. |
| P0 | Exporter không hỗ trợ checkpoint Dict | Lỗi đã tái lập ở `scripts/export_onnx.py:36`. Cần flatten/concat theo đúng thứ tự features extractor và bảo toàn activation. |
| P1 | Quantization chưa hỗ trợ model mới | Quantizer chặn input ngoài 6/39. Calibration tạo dữ liệu legacy, activation scale cố định; biến calibration được tạo nhưng chưa dùng để hiệu chỉnh dải activation. Chưa có kiểm chứng int8 cho policy mới. |
| P1 | Tần số/action chưa khớp | Desk train 20 Hz; RL task firmware 10 Hz. Sim pan/tilt 0,8/0,6 rad/s; adapter cũ 35/20 độ/s. Mô phỏng nhận m/s và rad/s, firmware trộn trực tiếp ra PWM. |
| P1 | Lệnh xích chưa đi qua RL task tới motor | `t_rax_board.cc:67–70` chỉ ghi target pan/tilt vào VisionReceiver. Giá trị motor trong ExecutionCommand chưa được áp dụng ở nhánh này. Các nhánh chase/choreography có đường motor riêng. |
| P1 | Chưa có một trọng tài cuối cho mọi lệnh | Idle, tracking, choreography và RL có thể tác động ở các đường khác nhau. Safety trong `AutonomousIdlePolicy::Update` không chứng minh mọi lệnh của các đường còn lại đã qua cùng bộ lọc. |
| P1 | Dữ liệu vision cũ chưa được xử lý nhất quán | Receiver có age phục vụ tracking, nhưng WorldState không tự xóa `target_visible` khi mất toàn bộ packet. Cần tuổi riêng cho từng nguồn, timeout và dừng khi dữ liệu cần thiết mất hiệu lực. |

Nguồn chính: `scripts/export_onnx.py`, `scripts/quantize_ppq.py`, `trax_rl/envs/trax_desk_env.py`, `trax_rl/envs/trax_desk_social_env.py`; bên firmware: `world_state.h/.cc`, `policy_runtime.h/.cc`, `autonomous_idle_policy.cc`, `t_rax_board.cc`, `idle_behavior.cc`, `safety_supervisor.cc`; bên vision: `app_main.cpp`, `trax_vision_protocol.h`.

## 3. Contract quan sát phải chuyển giao

Actor dùng Dict `desk_social_rl_v1`, tổng 71 số. Thứ tự flatten kiểm tra trực tiếp từ checkpoint:

| Trường | Offset trong vector flatten, tính từ 0 | Nguồn cần trên robot |
|---|---|---|
| elevation | 0–4 | Góc dọc camera cho ba người và hai vật đang thấy; cần calibration camera và mask khi mất target |
| humans | 5–28 | Ba slot ID bền vững: visible, góc ngang, range, ngồi, đi, vẫy, với/chỉ, khoảng cách bàn tay |
| memory | 29–33 | Ba cờ đã nhìn nhận người và hai cờ đã kiểm tra vật, gắn đúng ID, dwell và quy tắc reset |
| objects | 34–45 | Hai slot: visible, góc, range, đang cầm, vx/vy metric; cần bù chuyển động camera |
| robot | 46–55 | Vận tốc thân, yaw rate, pan/tilt, roll/pitch, lệnh dải xích và lệnh trước |
| sensors | 56–60 | ToF phía trước và bốn probe hỗ trợ mặt bàn |
| target | 61–64 | Bóng mục tiêu: visible, bearing, khoảng cách, kích thước tương đối |
| workspace | 65–70 | Pose chuẩn hóa, sin/cos yaw, khoảng đệm vùng/mép bàn |

Cùng 71 số nhưng sai thứ tự vẫn là input sai. Bbox pixel [-1,1] không tương đương bearing/π, PWM không tương đương m/s, diện tích bbox không phải khoảng cách metric. `current_pan_deg/current_tilt_deg` trong MotionController là giá trị đã ra lệnh, chưa phải feedback góc đo độc lập.

Định danh trong sim cố định. Nếu robot sắp ba người theo khoảng cách ở mỗi frame rồi đổi slot, memory sẽ gán sang người khác. Cần tracking ID, quy tắc tái sử dụng slot, timeout, confidence và reset theo thực thể. Hai slot vật cố định cũng chưa biểu diễn khái niệm phát hiện mọi đồ vật mới bất kỳ.

Không thể lấp các trường thiếu bằng 0 rồi gọi đó là triển khai tương thích. Hai hướng hợp lệ:

1. **Giữ nguyên actor 71D:** bổ sung perception đa mục tiêu, ước lượng metric, chuyển động/pose và cảm biến/safety cần thiết; tái tạo đúng contract rồi đánh giá. Camera ngoài nhìn marker trên bàn có thể cung cấp pose trong giai đoạn lab; IMU một mình không cung cấp x/y, encoder xích vẫn trôi khi trượt.
2. **Policy phù hợp phần cứng:** giữ checkpoint mới làm mốc tham chiếu/teacher, định nghĩa observation thực sự đo được, rồi train/fine-tune hoặc distill sang policy có contract mới. Muốn giữ nguyên năng lực ba người/hai vật vẫn phải bổ sung dữ liệu vision tương ứng; đổi observation không tự tạo ra thông tin cảm biến còn thiếu.

Với phần cứng hiện tại, ưu tiên hướng thứ hai cho một bản head-only có thân cố định. Một số trạng thái thân khi đó được xác định bằng gá thử đã hiệu chuẩn, không được tiếp tục dùng như hằng số khi cho xích chạy tự do. Loại hai action thân khỏi quyền thực thi làm thay đổi vòng kín, nên phải eval/train cấu hình khóa thân riêng.

Privileged reward shaping trong lúc train không tự làm policy không thể triển khai. Điểm chặn là actor đang phụ thuộc những quan sát chưa có nguồn thật; reward/goal scheduler của simulator không cần port nguyên xi sang firmware. Bộ nhớ và preprocessing cần một implementation độc lập MuJoCo.

## 4. Kiến trúc triển khai đề xuất

```text
Vision + cảm biến thật + timestamp/độ tin cậy
              ↓
Tracking + State estimator + Observation builder có version
              ↓
Actor đã kiểm chứng số học
              ↓
Trọng tài quyền điều khiển + giới hạn tốc độ/gia tốc
              ↓
Safety độc lập: timeout, dữ liệu lỗi, mép bàn, vật cản, dừng tay
              ↓
Motor/servo driver
```

Logging phải ghi được snapshot đầu vào, actor mean, action đã clip, lệnh sau safety, lệnh phần cứng và tuổi dữ liệu. Một đường cuối sở hữu cơ cấu chấp hành; các macro, tracking và RL cùng đi qua đó. Khóa motor trong head-only phải có hiệu lực với tất cả nguồn lệnh, không chỉ đặt action tiến/lùi của actor bằng 0.

Mạng nhỏ nên ưu tiên kiểm chứng đúng số học trước tối ưu int8. Có thể làm tham chiếu FP32 trên máy tính/gá lab, sau đó chọn runtime giữ đúng hai Tanh và output Linear + clip. Với ESP-DL, dùng công cụ quantization chính thức và đúng target, calibration bằng trạng thái thực đại diện, export test I/O rồi so output trên chip. Không chỉ mở giới hạn input và thay header của runtime ReLU cũ.

Tài liệu chính thức: [SB3 export](https://stable-baselines3.readthedocs.io/en/master/guide/export.html), [ESP-DL quantization và test I/O](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html), [operator support](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md). Cần chốt version ESP-IDF/ESP-DL thực dùng trước khi chọn backend.

## 5. Thứ tự thực hiện và điều kiện chuyển bước

Các ngưỡng dưới đây là đề xuất nghiệm thu, chưa phải kết quả đo trên robot.

| Bước | Công việc | Điều kiện chuyển bước |
|---|---|---|
| 1 — Contract và export | Inventory cảm biến; khóa schema, units, dấu, FOV, thứ tự slot; export actor không đổi activation | PyTorch deterministic và FP32 export sai khác tối đa ≤1e-5 trên bộ input đại diện; input sai shape/version/NaN bị từ chối |
| 2 — Dữ liệu thật, chưa cấp quyền motor | Thu vision/cảm biến, tracking IDs, tuổi dữ liệu; so sim/real feature distribution và replay | Không thiếu nguồn cho input được khai báo; lỗi/mất ID, che khuất, mất kết nối được xử lý và ghi log; không dùng dữ liệu simulator để giả làm dữ liệu thật |
| 3 — Head-only có gá | Cố định thân, khóa nguồn/quyền motor; eval/train head-only; áp pan/tilt qua trọng tài duy nhất | Không vượt cơ khí, không đổi dấu sai, không giật vì handoff; mất input thì về trạng thái xác định; log phản ứng xã hội trên người thật |
| 4 — Xích trong vùng thử có chặn cơ khí | Bật/kiểm chứng ToF, thêm chống rơi trước khi chạy bàn không có chặn; đo PWM–v, deadband, brake, trượt và nguồn điện; kiểm chứng watchdog | Tất cả nguồn lệnh chịu safety; ngắt nguồn dữ liệu/treo actor/dừng khẩn đều dừng; có đo thời gian dừng và quãng dừng. Tốc độ chỉ tăng theo mức đã đo được |
| 5 — Chuyển giao sim-to-real | Randomize theo dữ liệu đo: motor/servo, ma sát, trễ, FOV, miss detection, ID switch, ánh sáng và vị trí sự kiện | Nhiều bố trí/lịch sự kiện khác nhau và thử dài hạn; phản ứng từng người/từng vật, false response, số lần can thiệp, thời gian hoàn tất và mất target đều được báo cáo |
| 6 — Chạy tự do trên bàn | Có bảo vệ mép bàn và định vị/ranh giới tương ứng, bản rollback, test nguồn điện/audio/Wi-Fi đồng thời | Loop 20Hz nằm trong ngân sách 50ms ở điều kiện tải kiểm tra, có đo p95/p99; thời gian dừng và khoảng đệm phù hợp tốc độ thật; nghiệm thu phần cứng trước mở quyền |

Nếu chưa bổ sung chống rơi, giới hạn thử xích ở vùng có chặn cơ khí phù hợp và giám sát. Không coi thử trong vùng chặn là bằng chứng chạy tự do an toàn trên mép bàn. Giảm PWM không thay thế cảm biến mép và ngắt motor khi dữ liệu lỗi.

Không có cơ sở hứa thời gian hoàn tất port hoặc latency S3 trước khi benchmark. Cần ưu tiên đo được và tương thích contract trước khi train thêm đơn thuần; thêm nhiều bước RL trong scene hiện tại không sửa được các điểm chặn tích hợp ở trên.

## Việc nên làm ngay

Tạo nhánh triển khai riêng cho **observation thực tế + export giữ Tanh + chế độ head-only có khóa motor**, giữ nguyên checkpoint desk hiện tại làm đối chứng. Chỉ mở rộng sang xích sau khi có đường đo chuyển động và lớp chống rơi phù hợp.
