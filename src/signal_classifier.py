import cv2
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, List

class TrafficLightClassifier:
    """
    Classifies traffic signal states (RED, YELLOW, GREEN) from a specified ROI
    using HSV color segmentation and temporal smoothing across frames.
    """
    def __init__(self, roi: Tuple[int, int, int, int], history_len: int = 7, min_red_frames: int = 3, threshold: float = 0.003):
        self.roi = roi  # (x1, y1, x2, y2)
        self.threshold = threshold
        self.history = deque(maxlen=history_len)
        self.min_red_frames = min_red_frames

    def update_roi(self, new_roi: Tuple[int, int, int, int]):
        self.roi = new_roi

    def detect_state(self, frame: np.ndarray) -> Tuple[str, float, float, float]:
        """
        Extracts ROI from frame and detects current signal color state.
        """
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 'UNKNOWN', 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV hue boundary [0-10] and [170-180]
        red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([90, 255, 255]))

        red_score = float(np.count_nonzero(red_mask)) / red_mask.size
        yellow_score = float(np.count_nonzero(yellow_mask)) / yellow_mask.size
        green_score = float(np.count_nonzero(green_mask)) / green_mask.size

        if red_score > green_score and red_score > yellow_score and red_score > self.threshold:
            state = 'RED'
        elif yellow_score > red_score and yellow_score > green_score and yellow_score > self.threshold:
            state = 'YELLOW'
        elif green_score > red_score and green_score > yellow_score and green_score > self.threshold:
            state = 'GREEN'
        else:
            state = 'UNKNOWN'

        return state, red_score, yellow_score, green_score

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Processes frame, updates history, and returns current state with red confirmation status.
        """
        signal, r, y, g = self.detect_state(frame)
        entry = {
            'frame': frame_number,
            'signal': signal,
            'red_score': r,
            'yellow_score': y,
            'green_score': g
        }
        self.history.append(entry)

        recent_signals = [item['signal'] for item in self.history]
        red_confirmed = recent_signals.count('RED') >= self.min_red_frames

        return {
            'current_signal': signal,
            'red_confirmed': red_confirmed,
            'scores': (r, y, g),
            'history': list(self.history)
        }
