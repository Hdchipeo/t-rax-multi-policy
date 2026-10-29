import numpy as np
from typing import List, Dict, Any, Optional

class EventScheduler:
    """
    Dynamic Room Event Scheduler (Doc 02 - MuJoCo Environment).
    Sinh các kích thích môi trường ngẫu nhiên theo thời gian trong suốt episode:
    - sound_cue: Âm thanh bất ngờ tại một góc phòng (kích thích quay đầu)
    - human_move: Người di chuyển sang vị trí mới
    - ball_roll: Bóng hoặc vật thể lăn ngang qua trường nhìn
    - sudden_motion: Chuyển động đột ngột
    """
    def __init__(self,
                 mean_interval_sec: float = 8.0,
                 min_interval_sec: float = 3.0):
        self.mean_interval_sec = mean_interval_sec
        self.min_interval_sec = min_interval_sec
        self.rng = np.random.default_rng()

        self.scheduled_events: List[Dict[str, Any]] = []
        self.active_event: Optional[Dict[str, Any]] = None
        self.active_event_timer: float = 0.0

    def generate_episode_events(self, episode_length_sec: float = 50.0):
        """Lập lịch chuỗi sự kiện ngẫu nhiên cho toàn bộ episode"""
        self.scheduled_events.clear()
        self.active_event = None
        self.active_event_timer = 0.0

        current_time = float(self.rng.uniform(2.0, 5.0))
        event_types = ["sound_cue", "human_move", "ball_roll"]

        while current_time < episode_length_sec:
            etype = str(self.rng.choice(event_types))
            angle = float(self.rng.uniform(-np.pi * 0.6, np.pi * 0.6))
            distance = float(self.rng.uniform(0.8, 1.45))
            x = float(np.sin(angle) * distance)
            y = float(np.cos(angle) * distance)

            if etype == "ball_roll":
                roll_dir = 1.0 if self.rng.random() > 0.5 else -1.0
                y_pos = float(self.rng.uniform(0.8, 1.4))
                start_x = -1.2 * roll_dir
                end_x = 1.2 * roll_dir
                duration = float(self.rng.uniform(2.5, 4.0))
                start_p = np.array([start_x, y_pos, 0.08], dtype=np.float32)
                end_p = np.array([end_x, y_pos + float(self.rng.uniform(-0.2, 0.2)), 0.08], dtype=np.float32)
                self.scheduled_events.append({
                    "time": current_time,
                    "type": "ball_roll",
                    "start_pos": start_p,
                    "end_pos": end_p,
                    "pos": start_p.copy(),
                    "current_pos": start_p.copy(),
                    "intensity": float(self.rng.uniform(0.6, 1.0)),
                    "duration": duration,
                    "triggered": False
                })
            elif etype == "human_move":
                # Avatar người đứng/ngồi trực tiếp trên mặt sàn Z=0.0, cách xa tường bao Y=2.02m
                hx = float(np.clip(x, -1.2, 1.2))
                hy = float(np.clip(y, 0.8, 1.45))
                self.scheduled_events.append({
                    "time": current_time,
                    "type": "human_move",
                    "pos": np.array([hx, hy, 0.0], dtype=np.float32),
                    "intensity": float(self.rng.uniform(0.5, 1.0)),
                    "duration": float(self.rng.uniform(2.0, 4.0)),
                    "triggered": False
                })
            else: # sound_cue
                sx = float(np.clip(x, -1.4, 1.4))
                sy = float(np.clip(y, 0.6, 1.5))
                self.scheduled_events.append({
                    "time": current_time,
                    "type": "sound_cue",
                    "pos": np.array([sx, sy, 0.25], dtype=np.float32),
                    "intensity": float(self.rng.uniform(0.4, 1.0)),
                    "duration": float(self.rng.uniform(1.0, 2.5)),
                    "triggered": False
                })

            # Khoảng cách giữa các sự kiện tuân theo phân phối Poisson/Exponential
            interval = float(self.rng.exponential(self.mean_interval_sec))
            current_time += max(self.min_interval_sec, interval)

    def update(self, current_sim_time: float, dt: float) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra và kích hoạt sự kiện khi chạm mốc thời gian.
        Hỗ trợ nội suy vị trí bóng lăn thời gian thực (Continuous Rolling Ball Dynamics).
        """
        # Hết hạn sự kiện đang active hoặc cập nhật vị trí bóng lăn
        if self.active_event is not None:
            self.active_event_timer -= dt
            if self.active_event_timer <= 0:
                self.active_event = None
            elif self.active_event["type"] == "ball_roll" and "start_pos" in self.active_event:
                duration = max(0.1, self.active_event["duration"])
                progress = float(np.clip(1.0 - (self.active_event_timer / duration), 0.0, 1.0))
                self.active_event["current_pos"] = (1.0 - progress) * self.active_event["start_pos"] + progress * self.active_event["end_pos"]
                self.active_event["pos"] = self.active_event["current_pos"]

        # Quét các sự kiện sắp diễn ra
        for ev in self.scheduled_events:
            if not ev["triggered"] and current_sim_time >= ev["time"]:
                ev["triggered"] = True
                self.active_event = ev
                self.active_event_timer = ev["duration"]
                if ev["type"] == "ball_roll" and "start_pos" in ev:
                    ev["current_pos"] = ev["start_pos"].copy()
                    ev["pos"] = ev["start_pos"].copy()
                return ev

        # Emit only on the rising edge. Re-emitting an active sound/ball event
        # every control tick would continually retrigger it or reset velocity.
        return None
