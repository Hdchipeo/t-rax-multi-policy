# Người và sự kiện trên bàn T-Rax

Chạy từ thư mục dự án:

```sh
.venv/bin/python scripts/run_desk.py
.venv/bin/python scripts/run_desk.py --camera robot_closeup
.venv/bin/python scripts/run_desk.py --headless --steps 1800 --report reports/desk_environment/humans_smoke.json
.venv/bin/python scripts/run_desk.py --render reports/desk_environment/social_overview.png --render-time 44
.venv/bin/python scripts/run_desk.py --no-humans
.venv/bin/python scripts/run_desk.py --test
```

Mặc định mở camera `social_overview`, đủ rộng để xem bàn và người; dùng chuột để đổi góc nhìn trong viewer. Một episode dài 90 giây. Viewer lặp episode; headless chạy một episode. `--manual` giữ robot đứng yên để quan sát người. `--render-time` chạy mô phỏng đến thời điểm mong muốn trước khi chụp ảnh.

## Kịch bản mặc định

| Thời điểm (giây) | Tương tác |
|---|---|
| 0–5 | Người 1 đi tới ghế và ngồi xuống |
| 5–14 | Người 1 gõ bàn phím |
| 8–12 | Người 2 đi tới cạnh trái bàn |
| 12–17 | Người 2 rồi người 1 vẫy tay |
| 17–23 | Người 1 đưa ly nước tới bàn, thả ly ở giây 22 và rút tay |
| 23–28 | Người 1 gõ phím; người 2 chỉ về phía robot |
| 26–31 | Người 3 đi tới cạnh phải bàn; từ đây có ba người cùng hiện diện |
| 28–33 | Người 1 đưa tay lên tai như đang nghe điện thoại |
| 31–38 | Người 3 vẫy tay; người 2 đi quanh cạnh trái bàn |
| 34–39 | Người 1 tiếp tục gõ phím |
| 38–44 | Người 3 đặt khối đồ chơi vàng mới vào vùng T-Rax, thả ở giây 43 |
| 44–50 | Người 3 chỉ về phía robot; người 1 vẫy tay |
| 53–64 | Người 2 và người 3 lần lượt rời đi |
| 66–80 | Người 1 đứng lên, đi sang trái rồi rời bàn |
| 80–90 | Bàn vắng người; các vật đã đặt vẫn còn trên bàn |

Mỗi người có màu áo riêng, đầu, thân, hai tay và hai chân; có chuyển tiếp ngồi/đứng, bước chân, cử động tay gõ phím và vẫy. Nội suy chuyển động giúp tay đưa vật đến bề mặt trước khi thả. Khi bật `domain_randomization`, cả lịch dịch ngẫu nhiên 0–2 giây theo seed. Hiện chưa ngẫu nhiên hóa từng hành vi hoặc học chính sách con người.

## Vật lý và giới hạn

Người và ghế là hình học kinematic phục vụ tương tác, không có động lực học người, lực ghế hoặc va chạm cơ thể. Tay dùng quỹ đạo mục tiêu và IK đơn giản, giới hạn tầm với 59 cm; người nghiêng thân khi đưa vật tới bàn. Điện thoại chỉ được biểu diễn bằng cử chỉ tay lên tai, chưa có prop điện thoại.

Ly và đồ chơi là free bodies. Khi đang cầm, chúng theo tay và tắt va chạm; khi thả, trọng lực, ma sát và tiếp xúc bàn hoạt động. Ly được xấp xỉ bằng trụ rắn, phần nước là hình học trang trí; không mô phỏng dòng chảy hay nước đổ. Đồ chơi có thể bị robot đẩy nếu tiếp xúc. Chúng được ẩn, khóa vị trí ngoài cảnh và tắt va chạm trước sự kiện; reset xóa sạch trạng thái đặt vật và lịch sử sự kiện.

Bảo vệ `HUMAN_HAND_NEAR` dừng cả hai xích nếu tay gần tâm thân dưới 14 cm. Bảo vệ này dùng vị trí thật trong simulator, không phải khả năng phát hiện tay đã có trên robot thật. Geofence và ToF vẫn hoạt động như trước.

Demo có phản ứng theo kịch bản: tạm dừng và hướng đầu về người khi có sự kiện đến, vẫy tay hoặc thả đồ vật, rồi tiếp tục lộ trình. Nó sử dụng thông tin sự kiện mô phỏng và bị giới hạn góc servo; không đảm bảo luôn nhìn được người và chưa phải hành vi xã hội đã học bằng RL.

## Quan sát RL `desk_social_dict_v3`

Giữ các key `robot` (10), `target` (4), `sensors` (5), `workspace` (6), thêm:

| Key | Shape | Các cột theo thứ tự |
|---|---|---|
| `humans` | 3 × 8 | visible, bearing/π, range/2.5m, seated, walking, waving, reaching/pointing, khoảng cách tay gần nhất của người đó tới robot (clip 1m) |
| `objects` | 2 × 6 | visible, bearing/π, range/2.5m, held, vx, vy (m/s, clip ±1) |

Hàng người tương ứng ID 1–3; hàng vật tương ứng ly nước và đồ chơi mới. Nếu chưa xuất hiện hoặc bị khuất/ngoài FOV, cả hàng bằng 0. Nhìn thấy người hiện yêu cầu đầu nằm trong FOV camera; đây là quy tắc tổng hợp đơn giản, chưa phát hiện người qua bàn tay/thân khi đầu bị khuất. FOV dùng hình vuông bảo thủ theo góc dọc camera, giới hạn 2,5m; ray-cast kiểm tra vật che giữa camera và mục tiêu. Không đưa lịch tương lai hoặc nhãn hành động sắp tới vào observation. Đặc trưng vẫn lấy từ trạng thái simulator, chưa có detector ảnh hoặc nhận dạng cử chỉ thực.

`info['human_scenario']` là dữ liệu chẩn đoán: `time_s`, danh sách người/trạng thái, trạng thái vật, các sự kiện mới của bước và tổng số sự kiện. `events` chỉ chứa sự kiện đã bắt đầu trong bước, mỗi sự kiện được phát đúng một lần. Sự kiện giây 0 được trả trong info của reset. `env.humans.events` lưu lịch sử trong episode. Log này có thể dùng đánh giá phản ứng nhưng không được ghép vào input policy như tín hiệu thị giác.

Checkpoint phòng và checkpoint desk v2 chưa có hai key mới, cần cấu hình/huấn luyện lại policy tương ứng. Trong `TraxDeskEnv` gốc, reward vẫn là baseline khám phá/an toàn. Lớp train PPO bên dưới bổ sung reward xã hội và checkpoint riêng; xem báo cáo đánh giá trước khi kết luận năng lực.

## Kiểm chứng

39 tests đạt: 31 tests cũ và 8 tests mới về chuỗi ba người, thả vật và tiếp xúc bàn, liên tục vị trí lúc thả, reset, seed, FOV/che khuất, dừng vì tay và Gym contract. Chạy demo 1.800 bước (90 giây), có đủ ba người, hai vật còn trên bàn, không va chạm/rơi/mất ổn định. Chi tiết: `reports/desk_environment/humans_smoke.json`.

Mã: `assets/trax/desk_humans.xml`, `trax_rl/dynamics/desk_humans.py`, tích hợp trong `trax_rl/envs/trax_desk_env.py` và `scripts/run_desk.py`.

## Train RL trên ngữ cảnh này

Đã thêm pipeline PPO và contract mở rộng `desk_social_rl_v1`. Xem [hướng dẫn huấn luyện, reward và đánh giá](desk_social_rl.md). Chạy `scripts/run_desk.py --policy <checkpoint.zip>` để xem mạng đã học; chạy không có `--policy` vẫn là demo theo kịch bản.
