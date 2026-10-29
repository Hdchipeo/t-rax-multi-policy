import abc
import numpy as np
from typing import Dict, Any, Optional

class BasePolicy(abc.ABC):
    """
    Abstract Base Class cho các chuyên gia Policy trong kiến trúc Multi-Policy RL của T-RAX.
    Mỗi Policy tập trung giải quyết DUY NHẤT một hành vi chuyên biệt:
      - ExplorePolicy: Khám phá không gian với duy nhất 1 cảm biến ToF phía trước
      - LookAroundPolicy: Quét và đảo mắt tìm kiếm kích thích thị giác
      - ApproachPolicy: Tiếp cận mục tiêu và điều hòa cự ly dừng an toàn
      - ThreatEscapePolicy: Thoát hiểm khẩn cấp khi gặp đe dọa / va đập
    """
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        """Khởi tạo lại bộ đệm trạng thái nội tại của Policy giữa các episode"""
        pass

    @abc.abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Sinh vector hành động 5 chiều:
        [linear_velocity, angular_velocity, pan_velocity, tilt_velocity, attention_intensity]
        """
        raise NotImplementedError
