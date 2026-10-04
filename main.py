import os
import argparse
import yaml
import cv2
from collections import defaultdict, deque
from pathlib import Path

from src import (
    TrafficLightClassifier,
    LineCrossingDetector,
    VehicleTracker,
    PlateRecognizer,
    ViolationLogger
)
from utils import draw_pipeline_ui

def main():
    parser = argparse.ArgumentParser(description="Traffic Signal Violation Detection System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file or webcam index (0)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window display during processing")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize modules
    print("[INFO] Initializing Vehicle Tracker (YOLOv8 + ByteTrack)...")
    tracker = VehicleTracker(
        model_path=config['models']['vehicle_model'],
        conf=config['models']['confidence'],
        imgsz=config['models']['imgsz'],
        classes=config['vehicle_classes']
    )

    print("[INFO] Initializing Traffic Signal Classifier...")
    signal_calib = config['signal_calibration']
    signal_classifier = TrafficLightClassifier(
        roi=tuple(signal_calib['traffic_light_roi']),
        history_len=signal_calib['signal_history_len'],
        min_red_frames=signal_calib['min_red_frames'],
        threshold=signal_calib['color_threshold']
    )

    print("[INFO] Initializing Line Crossing Detector...")
    rules = config['violation_rules']
    line_detector = LineCrossingDetector(
        stop_line=(tuple(signal_calib['stop_line'][0]), tuple(signal_calib['stop_line'][1])),
        pre_cross_margin=rules['pre_cross_margin'],
        post_cross_margin=rules['post_cross_margin'],
        max_crossing_jump=rules['max_crossing_jump'],
        min_downward_motion=rules['min_downward_motion']
    )

    print("[INFO] Initializing License Plate Recognizer...")
    plate_recognizer = PlateRecognizer(use_easyocr=True)

    print("[INFO] Initializing Violation Logger...")
    out_cfg = config['output']
    logger = ViolationLogger(
        json_path=out_cfg['output_json'],
        snapshot_dir=out_cfg['snapshot_dir'],
        save_snapshots=out_cfg['save_snapshots']
    )

    # Video Setup
    video_source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = out_cfg['output_video']
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Tracking storage
    track_history = defaultdict(lambda: deque(maxlen=30))
    track_seen_frames = defaultdict(int)
    flagged_ids = set()

    print(f"[INFO] Processing video: {args.video} ({total_frames} frames)...")

    results_generator = tracker.track_stream(args.video)

    for frame_number, result in enumerate(results_generator):
        frame = result.orig_img.copy()

        # 1. Traffic signal state
        sig_info = signal_classifier.process_frame(frame, frame_number)
        current_signal = sig_info['current_signal']
        red_confirmed = sig_info['red_confirmed']

        vehicles_in_frame = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, ids, classes):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                bottom_center = ((x1 + x2) // 2, y2)
                class_name = tracker.class_names.get(class_id, "vehicle")

                vehicles_in_frame.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "bottom_center": bottom_center
                })

                history = track_history[track_id]
                previous_points = list(history)
                track_seen_frames[track_id] += 1
                history.append(bottom_center)

                # Skip if already logged or insufficient history
                if track_id in flagged_ids or track_seen_frames[track_id] < rules['min_track_frames'] or not previous_points:
                    continue

                previous_point = previous_points[-1]

                # Check line crossing event
                is_crossing, debug_info = line_detector.check_crossing(bottom_center, previous_point)

                # Flag violation if crossed when RED signal confirmed
                if is_crossing and red_confirmed:
                    flagged_ids.add(track_id)

                    # Recognize license plate
                    plate_info = plate_recognizer.recognize(frame, (x1, y1, x2, y2))

                    # Log violation
                    record = logger.log_violation(
                        frame_number=frame_number,
                        track_id=track_id,
                        vehicle_class=class_name,
                        bbox=[x1, y1, x2, y2],
                        signal_state=current_signal,
                        plate_info=plate_info,
                        frame=frame
                    )

                    print(f"🔥 [VIOLATION DETECTED] Frame {frame_number} | Vehicle #{track_id} ({class_name}) | Signal: {current_signal} | Plate: {plate_info['plate_text']}")

        # 2. Draw UI
        annotated_frame = draw_pipeline_ui(
            frame=frame,
            stop_line=(tuple(signal_calib['stop_line'][0]), tuple(signal_calib['stop_line'][1])),
            traffic_light_roi=tuple(signal_calib['traffic_light_roi']),
            signal_state=current_signal,
            vehicles=vehicles_in_frame,
            violation_ids=flagged_ids
        )

        writer.write(annotated_frame)

        if not args.no_display:
            cv2.imshow("Traffic Violation System", cv2.resize(annotated_frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"[COMPLETED] Pipeline execution finished.")
    print(f"  - Annotated video saved to: {output_video_path}")
    print(f"  - Violations JSON saved to: {out_cfg['output_json']}")
    print(f"  - Snapshots directory: {out_cfg['snapshot_dir']}")
    print(f"  - Total Violations Recorded: {len(flagged_ids)}")

if __name__ == "__main__":
    main()
