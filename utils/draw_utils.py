import cv2
import numpy as np
from typing import Tuple, List, Set, Dict, Any

def draw_pipeline_ui(frame: np.ndarray,
                     stop_line: Tuple[Tuple[int, int], Tuple[int, int]],
                     traffic_light_roi: Tuple[int, int, int, int],
                     signal_state: str,
                     vehicles: List[Dict[str, Any]],
                     violation_ids: Set[int]) -> np.ndarray:
    """
    Renders high-quality visual overlays:
    - Stop Line (Yellow / Red on Violation)
    - Traffic Light ROI Box & Badge (Green, Yellow, Red)
    - Vehicle Bounding Boxes & Track IDs
    - Violation Banners for flagged vehicles
    """
    annotated = frame.copy()

    # 1. Draw Traffic Light ROI & Signal Badge
    rx1, ry1, rx2, ry2 = traffic_light_roi
    signal_color = (0, 255, 0) if signal_state == 'GREEN' else ((0, 255, 255) if signal_state == 'YELLOW' else (0, 0, 255))
    
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), signal_color, 2)
    
    # Badge background
    badge_text = f"SIGNAL: {signal_state}"
    cv2.rectangle(annotated, (rx1, max(0, ry1 - 30)), (rx1 + 160, ry1), signal_color, -1)
    cv2.putText(annotated, badge_text, (rx1 + 5, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Draw Stop Line
    line_color = (0, 0, 255) if signal_state == 'RED' else (0, 255, 255)
    cv2.line(annotated, stop_line[0], stop_line[1], line_color, 4)
    cv2.putText(annotated, "STOP LINE", (stop_line[0][0] + 10, stop_line[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

    # 3. Draw Tracked Vehicles
    for v in vehicles:
        track_id = v['track_id']
        x1, y1, x2, y2 = v['bbox']
        class_name = v['class_name']
        bottom_center = ((x1 + x2) // 2, y2)

        is_violator = track_id in violation_ids
        box_color = (0, 0, 255) if is_violator else (0, 255, 0)

        # Draw vehicle bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(annotated, bottom_center, 5, (255, 0, 0), -1)

        # Label
        label = f"#{track_id} {class_name}"
        if is_violator:
            label += " [VIOLATION]"

        # Draw text background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + w + 10, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated
