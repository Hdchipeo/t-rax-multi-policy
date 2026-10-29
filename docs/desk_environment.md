# T-Rax: môi trường bàn làm việc MuJoCo

## Bố trí và kích thước

Bản dựng dùng mét, lấy kích thước ban đầu để mô phỏng; chưa đo bàn thực tế.

| Hạng mục | Kích thước/vị trí |
|---|---|
| Bàn | 1,40 × 0,75 m, mặt bàn cao 0,76 m |
| Vùng riêng T-Rax | 0,50 × 0,40 m, tâm (0,40; 0,02), phía phải bàn |
| Robot | CAD hiện có; footprint khoảng 0,11 × 0,10 m |
| Vị trí ban đầu | (0,30; -0,06), trên pad nghỉ |
| Màn hình | Phía sau/trái, khoảng 0,57 × 0,33 m |
| Bàn phím, chuột | Phía trước màn hình, ngoài vùng T-Rax |
| Trang trí | Cốc, sổ, chậu cây, vật trang trí nhỏ; bóng đường kính 3 cm trong vùng robot |

Hướng +Y là phía sau bàn; -Y là phía người ngồi. Màn hình có bezel, chân đế và giao diện giả lập; bàn phím có phím riêng. Các props có hình học va chạm. Vùng riêng là thảm và đường viền; không có tường giữ robot vô hình.

- Scene chính: `assets/trax/desk_scene.xml`.
- Gym environment: `trax_rl/envs/trax_desk_env.py`, class `TraxDeskEnv`.
- Cấu hình: `configs/environments/desk_v1.yaml`.
- Demo/render: `scripts/run_desk.py`.

## Chạy

Tại thư mục dự án:

```sh
# Mở toàn bộ bàn, chạy demo quan sát/di chuyển/chờ.
.venv/bin/python scripts/run_desk.py

# Nhìn gần vùng hoạt động, hoặc nhìn từ camera trên đầu robot.
.venv/bin/python scripts/run_desk.py --camera robot_closeup
.venv/bin/python scripts/run_desk.py --camera robot_camera

# Xem vật lý ở trạng thái đứng yên.
.venv/bin/python scripts/run_desk.py --manual

# Chạy kiểm tra 120 giây mô phỏng, không mở cửa sổ.
.venv/bin/python scripts/run_desk.py --headless --steps 2400 --report reports/desk_environment/smoke.json

# Chạy toàn bộ tests, có bootstrap MuJoCo cho ổ exFAT.
.venv/bin/python scripts/run_desk.py --test

# Render toàn cảnh ra PNG.
.venv/bin/python scripts/run_desk.py --render reports/desk_environment/overview.png
```

Trên macOS, script tự gọi mjpython cho native viewer, qua Python để không phụ thuộc shebang cũ sau khi di chuyển venv. Nếu exFAT tạo file AppleDouble `._*.dylib` khiến MuJoCo không import được, launcher dùng một bản cache sạch của gói MuJoCo trong thư mục tạm; không xóa hoặc sửa thư viện đã cài. Entry point này là đường chạy được hỗ trợ cho desktop scene.

Demo là bộ điều khiển theo bố trí đã biết: nghỉ → quét đầu → đi giữa các điểm trong vùng → nhìn bóng → quay về. Đây chưa phải policy RL được huấn luyện trên bàn.

## Vật lý và bảo vệ mép bàn

Đế T-Rax dùng freejoint 6 bậc tự do và hai dải tiếp xúc xích trái/phải liên tục. Đã bỏ bốn bánh tiếp xúc thay thế. Vận tốc bề mặt dưới dải xích được đặt bằng `geom_surfacevel` của MuJoCo; bộ giải tiếp xúc/ma sát tạo lực đẩy và mô-men quay, không ghi đè vị trí hoặc vận tốc thân. Hai dải cùng tốc độ giúp tiến/lùi; khác tốc độ tạo rẽ và chạy ngược chiều tạo quay tại chỗ. Có animation mắt xích chạy quanh hai vòng xích. Mặt bàn hữu hạn, có trọng lực và sàn dưới bàn: robot có thể ngã khi tắt bảo vệ.

Điều khiển 20 Hz, tích phân vật lý 500 Hz. Tốc độ tối đa 0,08 m/s và 1 rad/s; giới hạn gia tốc và khoảng đệm quanh footprint giúp robot dừng trong vùng riêng. Giới hạn bàn/vùng được đọc từ geometry MJCF, tránh tiếp tục dùng biên phòng ±1,9 m.

Geofence là bảo vệ bằng phần mềm dùng pose mô phỏng. Bốn ray nhìn xuống mô phỏng cảm biến mép bàn bổ sung; chúng **không phải cảm biến đã có trên T-Rax thật**. ToF phía trước dùng ray-cast vào geometry. Không suy diễn rằng một ToF phía trước đủ ngăn robot thật rơi khỏi bàn.

Khi cần thử nghiệm vật lý ngã rơi, dùng `TraxDeskEnv(safety_enabled=False, restrict_to_zone=False)` trong test riêng. Mặc định giữ bảo vệ bật. Đây là mô hình xích liên tục rút gọn, chưa mô phỏng từng mắt xích bằng khớp riêng, lực căng dây xích hoặc động lực điện của motor. Vận tốc dải xích là giá trị áp đặt, không phải encoder motor đo được; thông số ma sát chưa được hiệu chuẩn với robot thật. Mô hình cần MuJoCo >= 3.11. Tham khảo [surfacevel trong tài liệu MuJoCo](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom-surfacevel).

## Contract RL riêng

`action_space`: Box 5 chiều [-1,1] cho tiến/lùi, quay thân, vận tốc pan, vận tốc tilt và intensity. Kênh intensity hiện chỉ là placeholder cho nhiệm vụ biểu cảm, chưa có LED/âm thanh vật lý.

`observation_space`: Dict `desk_social_dict_v3`:

| Key | Dim | Nội dung |
|---|---:|---|
| robot | 10 | Vận tốc thân, pan/tilt, roll/pitch, vận tốc áp đặt của hai dải xích, lệnh trước |
| target | 4 | Bóng có trong FOV, sai số hướng nhìn, cự ly, kích thước tương đối |
| sensors | 5 | ToF phía trước và 4 probe hỗ trợ mặt bàn |
| workspace | 6 | Pose trong vùng, sin/cos yaw, khoảng đệm vùng/mép bàn |
| humans | 3 × 8 | Người trong tầm nhìn, vị trí tương đối và trạng thái tương tác |
| objects | 2 × 6 | Ly nước/đồ vật mới trong tầm nhìn, trạng thái cầm và vận tốc |

Observation hiện được tổng hợp từ simulator; chưa có detector thị giác học từ ảnh. Có camera toàn cảnh, cận cảnh, từ trên xuống và camera gắn đầu robot. Muốn huấn luyện PPO có thể dùng `MultiInputPolicy` với Dict này. Checkpoint phòng 39D/54D cũ không tương thích trực tiếp với desk contract và khoảng cách centimet; không dùng kết quả pass phòng cũ để kết luận khả năng trên bàn.

Reward trong env là baseline exploration/energy/contact/geofence để kiểm tra kết nối Gym, chưa phải reward hoàn chỉnh cho các nhiệm vụ xã hội.

## Kiểm chứng ngày 15/10/2026

- Scene nạp thành công: 2 actuator pan/tilt; hai kênh vận tốc dải xích được đưa vào bộ giải tiếp xúc qua `geom_surfacevel`. Không còn joint/actuator bánh tiếp xúc thay thế.
- Gym checker đạt.
- 14 test desktop: có hai dải tiếp xúc và không có joint bánh thay thế; xích cùng chiều/ngược chiều; reset tốc độ; animation mắt xích; xích trên không không tạo lực đẩy thân; cùng các test bố trí, support, mép bàn và va chạm.
- Toàn bộ 31 test trong dự án đạt, gồm 17 test behavioral trước đó.
- Demo 2.400 bước = 120 giây mô phỏng: hoàn thành 2 vòng, không va chạm, không rơi, không cảnh báo mất ổn định. Chi tiết khoảng đệm và số vòng ở `reports/desk_environment/smoke.json` (được chạy lại sau khi đổi sang xích liên tục).
- Đã render và kiểm tra ảnh toàn cảnh/cận cảnh trong `reports/desk_environment/`.

## Người và sự kiện tương tác

Mặc định có kịch bản 90 giây với ba người đến/ngồi/di chuyển/rời bàn, vẫy tay, gõ phím, chỉ về phía robot, nghe điện thoại và đặt ly nước/đồ chơi. Camera mặc định là `social_overview`. Xem [lịch sự kiện, contract quan sát và giới hạn](desk_human_scenarios.md). Kiểm chứng mới: 39 tests đạt; lượt demo 90 giây không va chạm, không rơi hoặc mất ổn định.
