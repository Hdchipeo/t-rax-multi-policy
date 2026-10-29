"""
T-RAX Multi-Policy Architecture (Upgrade Plan Phase 3 & 4)
Modular, single-responsibility Reinforcement Learning & Behavioral Policies:
- ExplorePolicy: Locomotion exploration with single front ToF sensor
- LookAroundPolicy: Pan/Tilt head visual exploration
- ApproachPolicy: Target approach with front ToF distance regulation
- ThreatEscapePolicy: Emergency retreat & avoidance under high fear/threat
"""

from trax_rl.policies.base_policy import BasePolicy
from trax_rl.policies.explore_policy import ExplorePolicy
from trax_rl.policies.look_around_policy import LookAroundPolicy
from trax_rl.policies.approach_policy import ApproachPolicy
from trax_rl.policies.threat_escape_policy import ThreatEscapePolicy

__all__ = [
    "BasePolicy",
    "ExplorePolicy",
    "LookAroundPolicy",
    "ApproachPolicy",
    "ThreatEscapePolicy"
]
