import cv2
import numpy as np
import re
from typing import Optional, Dict, Any

class PlateRecognizer:
    """
    License Plate Recognition engine with perspective correction,
    image enhancement (CLAHE, denoising), and OCR text extraction.
    """
    def __init__(self, use_easyocr: bool = True):
        self.use_easyocr = use_easyocr
        self.ocr_reader = None

        if self.use_easyocr:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception:
                self.use_easyocr = False

    def perspective_correction(self, plate_crop: np.ndarray) -> np.ndarray:
        """Applies perspective transformation to straighten license plates."""
        if plate_crop.size == 0:
            return plate_crop
            
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return plate_crop

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Order box points: top-left, top-right, bottom-right, bottom-left
        pts = box[np.argsort(box[:, 1])]
        top = pts[:2][np.argsort(pts[:2, 0])]
        bottom = pts[2:][np.argsort(pts[2:, 0])]
        ordered_box = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

        w = int(max(np.linalg.norm(ordered_box[1] - ordered_box[0]), np.linalg.norm(ordered_box[2] - ordered_box[3])))
        h = int(max(np.linalg.norm(ordered_box[3] - ordered_box[0]), np.linalg.norm(ordered_box[2] - ordered_box[1])))

        if w <= 0 or h <= 0:
            return plate_crop

        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
        warped = cv2.warpPerspective(plate_crop, M, (w, h))

        return warped

    def enhance_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Upscales, applies CLAHE contrast enhancement, and denoises plate images."""
        if plate_crop.size == 0:
            return plate_crop

        resized = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return denoised

    def clean_text(self, text: str) -> str:
        """Standardizes alphanumeric license plate characters."""
        text = text.upper()
        clean = re.sub(r'[^A-Z0-9]', '', text)
        return clean

    def recognize(self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Extracts license plate from vehicle region, processes image, and reads license plate text.
        """
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]

        # Clamp box coordinates to image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return {"plate_text": "UNKNOWN", "confidence": 0.0}

        corrected = self.perspective_correction(vehicle_crop)
        enhanced = self.enhance_plate(corrected)

        plate_text = "UNKNOWN"
        confidence = 0.0

        if self.use_easyocr and self.ocr_reader is not None:
            try:
                ocr_out = self.ocr_reader.readtext(enhanced)
                if ocr_out:
                    best_match = max(ocr_out, key=lambda x: x[2])
                    raw_text, conf = best_match[1], float(best_match[2])
                    cleaned = self.clean_text(raw_text)
                    if len(cleaned) >= 3:
                        plate_text = cleaned
                        confidence = conf
            except Exception:
                pass

        return {
            "plate_text": plate_text,
            "confidence": round(confidence, 2)
        }
