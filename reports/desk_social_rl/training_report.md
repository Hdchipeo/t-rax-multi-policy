# Kết quả train RL bàn làm việc có người

Đã hoàn tất 524,288 transitions PPO: 262.144 bước khởi tạo mới và 262.144 bước fine-tune.
Checkpoint chọn bằng validation nằm ở `checkpoints/desk_social_ppo_v1_finetune_20261015_135543/best_model.zip`, tại bước 458,752.
Actor điều khiển trực tiếp; không sử dụng `DeskDemo` hay checkpoint phòng.

| Bộ điều khiển | Đáp nhìn vẫy tay | Người nhận biết / 3 | Vật quan sát / 2 | Safety override | Episode có va chạm/rơi/bất ổn |
|---|---:|---:|---:|---:|---:|
| Stationary | 0.0% | 0.00 | 0.33 | 0.00% | 0 / 3 |
| Random actions | 0.0% | 0.00 | 0.33 | 0.37% | 0 / 3 |
| Untrained PPO | 0.0% | 0.00 | 0.00 | 0.00% | 0 / 3 |
| First run best | 25.0% | 2.00 | 0.00 | 0.00% | 0 / 3 |
| Fine-tune best | 50.0% | 2.00 | 1.00 | 0.00% | 0 / 3 |
| Additional audit (5 seeds) | 65.0% | 2.00 | 1.00 | 0.00% | 0 / 5 |

Các hàng baseline và hai run dùng seed 2001–2003, yaw đầu 0, π, -π/2; mỗi episode dài 90 giây. Mọi bộ điều khiển đều có cùng lớp bảo vệ bật. Safety bằng 0 va chạm không chứng minh policy tự an toàn nếu bỏ bảo vệ.

Validation best đạt đủ gate: **True**. Test 3 seed đạt đủ gate: **True**.
Các gate test: `{"no_collisions_falls_or_instability": true, "waves": true, "people": true, "objects": true, "safety_overrides": true}`.

Gate thử nghiệm: không va chạm/rơi/bất ổn, đáp nhìn ≥50%, nhận biết trung bình ≥2 người, quan sát ≥1 vật và safety override ≤5%. Đây là mốc thử nghiệm trong scene cố định, chưa yêu cầu nhận biết đủ cả ba người hoặc kiểm tra cả hai vật trong mọi episode.

Audit bổ sung một lần trên seed 3001–3005, yaw 0, π, -π/2, π/2, 0,6: đạt đủ gate **True**. Audit không được dùng chọn checkpoint.

## Tái lập và giới hạn

Cấu hình và metadata của mỗi run ghi seed, hyperparameters, version thư viện, source SHA và checkpoint SHA. Bản sao runtime nguồn đã lưu trong `runtime_source/` của từng run.
Checkpoint SHA-256: `6f9e01ff65c839433248d738fd1982d7097a1544fc1e26f8692cfbc026a5f657`.
46 regression tests đạt; smoke train, resume, save/load và chạy policy headless đã kiểm tra.

Khác biệt seed chủ yếu là độ trễ toàn lịch 0–2 giây và vị trí bóng. Các kết quả chưa kiểm chứng tổng quát sang bố trí bàn/người mới. Người là avatar kinematic, nước là hình học trang trí, perception là tổng hợp từ MuJoCo. Reward và safety được phép dùng simulator ground truth, còn actor nhận FOV/occlusion và memory quan sát được. Chưa thử trên robot thật.

Đáp lại vẫy tay được đo bằng hướng nhìn và dwell, chưa phải lời nói, vẫy đáp hoặc tương tác hai chiều. `idle_fraction` đo thân gần đứng yên, không chứng minh đầu nghỉ hoặc hành vi tự nhiên theo đánh giá con người.

Checkpoint được giữ làm kết quả thực nghiệm; các gate chưa đạt cần tiếp tục xử lý thay vì suy ra năng lực từ reward tăng.

## Chạy checkpoint

```sh
.venv/bin/python scripts/run_desk.py --policy checkpoints/desk_social_ppo_v1_finetune_20261015_135543/best_model.zip
```

Lịch sử validation: `validation_history.csv`. Các lần có cùng số bước có thể là trước và sau cập nhật PPO của rollout cuối.
