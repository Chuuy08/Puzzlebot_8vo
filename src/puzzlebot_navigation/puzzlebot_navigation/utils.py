#!/usr/bin/env python3
import math
import numpy as np


def quat_to_yaw(q) -> float:
    """Extract yaw from a ROS quaternion message (geometry_msgs/Quaternion)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a: float) -> float:
    """Wrap a scalar angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def wrap_angle_v(a: np.ndarray) -> np.ndarray:
    """Wrap a numpy array of angles to [-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))
