import numpy as np
from typing import Tuple, List

class LineCrossingDetector:
    """
    Handles geometry calculations for vehicle stop line crossing,
    including signed distance from stop line and motion validation.
    """
    def __init__(self, stop_line: Tuple[Tuple[int, int], Tuple[int, int]], 
                 pre_cross_margin: float = 5.0, 
                 post_cross_margin: float = 0.0,
                 max_crossing_jump: float = 120.0,
                 min_downward_motion: float = 2.0):
        self.p1 = stop_line[0]  # (x1, y1)
        self.p2 = stop_line[1]  # (x2, y2)
        self.pre_cross_margin = pre_cross_margin
        self.post_cross_margin = post_cross_margin
        self.max_crossing_jump = max_crossing_jump
        self.min_downward_motion = min_downward_motion

    def line_y_at_x(self, x: float) -> float:
        """Calculates Y coordinate of stop line at a given X coordinate."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        if x2 == x1:
            return float(min(y1, y2))
        return float(y1 + (y2 - y1) * ((x - x1) / (x2 - x1)))

    def signed_line_distance(self, point: Tuple[int, int]) -> float:
        """
        Calculates signed perpendicular/vertical distance from stop line.
        Negative (< 0): Vehicle is before/above the stop line.
        Positive (>= 0): Vehicle is on/after the stop line.
        """
        x, y = point
        return float(y - self.line_y_at_x(x))

    def check_crossing(self, current_point: Tuple[int, int], previous_point: Tuple[int, int]) -> Tuple[bool, dict]:
        """
        Checks if a vehicle moved from before the line to past the line in a valid manner.
        """
        current_dist = self.signed_line_distance(current_point)
        previous_dist = self.signed_line_distance(previous_point)

        downward_motion = current_point[1] - previous_point[1]
        distance_jump = current_dist - previous_dist

        was_before_line = (previous_dist <= -self.pre_cross_margin)
        crossed_line = (current_dist >= self.post_cross_margin)
        reasonable_jump = (distance_jump <= self.max_crossing_jump)
        moving_forward = (downward_motion >= self.min_downward_motion)

        is_crossing = (was_before_line and crossed_line and reasonable_jump and moving_forward)

        info = {
            "was_before_line": was_before_line,
            "crossed_line": crossed_line,
            "reasonable_jump": reasonable_jump,
            "moving_forward": moving_forward,
            "current_dist": current_dist,
            "previous_dist": previous_dist,
            "downward_motion": downward_motion
        }

        return is_crossing, info
