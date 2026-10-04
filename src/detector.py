import os
from typing import List, Dict, Generator, Any
import numpy as np
from ultralytics import YOLO

class VehicleTracker:
    """
    Wrapper around YOLOv8 model and ByteTrack tracker for detecting and
    tracking vehicles across video frames.
    """
    def __init__(self, model_path: str = "best.pt", conf: float = 0.25, imgsz: int = 1280, classes: List[int] = [2, 3, 4]):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model file not found at: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.class_names = self.model.names

    def track_stream(self, source_path: str) -> Generator[Any, None, None]:
        """
        Yields tracking result generators frame by frame using ByteTrack.
        """
        results = self.model.track(
            source=source_path,
            tracker='bytetrack.yaml',
            stream=True,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False
        )
        return results
