import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

class ViolationLogger:
    """
    Saves violation records to structured JSON files and outputs annotated image snapshots.
    """
    def __init__(self, json_path: str = "violations.json", snapshot_dir: str = "violations_snapshots", save_snapshots: bool = True):
        self.json_path = json_path
        self.snapshot_dir = snapshot_dir
        self.save_snapshots = save_snapshots
        self.records: List[Dict[str, Any]] = []

        if self.save_snapshots:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def log_violation(self, frame_number: int, track_id: int, vehicle_class: str,
                      bbox: List[int], signal_state: str, plate_info: Dict[str, Any],
                      frame: np.ndarray) -> Dict[str, Any]:
        """
        Logs a red-light violation event and saves a frame snapshot.
        """
        timestamp = datetime.now().isoformat()
        snapshot_filename = f"violation_track_{track_id}_frame_{frame_number}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)

        if self.save_snapshots and frame is not None:
            # Save cropped or annotated snapshot
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            margin = 30
            crop_x1, crop_y1 = max(0, x1 - margin), max(0, y1 - margin)
            crop_x2, crop_y2 = min(w, x2 + margin), min(h, y2 + margin)
            
            snapshot_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if snapshot_crop.size > 0:
                cv2.imwrite(snapshot_path, snapshot_crop)
            else:
                cv2.imwrite(snapshot_path, frame)

        record = {
            "violation_id": len(self.records) + 1,
            "timestamp": timestamp,
            "frame_number": frame_number,
            "vehicle_track_id": track_id,
            "vehicle_class": vehicle_class,
            "bbox": bbox,
            "signal_state": signal_state,
            "license_plate": plate_info.get("plate_text", "UNKNOWN"),
            "plate_confidence": plate_info.get("confidence", 0.0),
            "snapshot": snapshot_path if self.save_snapshots else None
        }

        self.records.append(record)
        self.save_json()
        return record

    def save_json(self):
        """Flushes records array to JSON file."""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
