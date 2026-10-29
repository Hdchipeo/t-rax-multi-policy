# Sửa cơ cấu truyền động bánh xích — 15/10/2026

Đã thay mô hình 4 bánh tiếp xúc bằng 2 dải tiếp xúc xích liên tục trái/phải. CAD thân/đầu được giữ nguyên. Tread bars chạy trên hai vòng xích theo vận tốc dải; không còn wheel joint, wheel velocity actuator hoặc cylinder bánh tiếp xúc thay thế.

- Tốc độ dải: `v_left = v - omega * 0.090 / 2`, `v_right = v + omega * 0.090 / 2`.
- Mặt dưới xích chạy ngược hướng tiến. `geom_surfacevel` đặt vận tốc vật liệu trong hệ tọa độ local của từng dải; MuJoCo giải ma sát/tiếp xúc để thân chuyển động.
- Pose, vận tốc thân và lực trọng trường do MuJoCo tích phân. Không ghi đè pose/vận tốc, không thêm lực tiến khi xích không tiếp xúc.
- Giữ freejoint, mặt bàn hữu hạn, geofence và kiểm tra rơi.
- MuJoCo tối thiểu 3.11, bản hiện tại đã kiểm tra là 3.14.0. [Tài liệu surfacevel](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom-surfacevel).
- Observation chuyển sang `desk_tracks_dict_v2`; slot vận tốc bánh trước đây nay là vận tốc áp đặt của hai dải xích, không phải encoder motor.

31/31 test đạt. Thêm các test xác nhận không có wheel proxy joints, hai dải cùng tốc độ đẩy tiến, hai dải ngược chiều tạo quay, reset vận tốc dải, animation và không có lực đẩy trong không khí. Gym checker đạt.

Demo 2.400 bước, 120 giây: 2 vòng, 0 contact va chạm ngoài mặt bàn, không rơi, không cảnh báo mất ổn định. Khoảng đệm nhỏ nhất so với biên tâm an toàn: 0,0291 m. Dữ liệu trong `tracks_smoke.json`.

Đây là mô hình xích liên tục rút gọn. Chưa mô phỏng từng mắt bằng khớp riêng, độ căng xích, bánh răng ăn khớp hoặc dòng điện/mô-men motor; chưa hiệu chuẩn ma sát với phần cứng. Không gọi đây là mô phỏng đầy đủ cơ cấu xích thật.
