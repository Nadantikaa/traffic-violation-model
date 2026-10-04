import cv2
import yaml
import numpy as np

def run_calibration_tool(video_path: str, frame_number: int = 100, config_out: str = "config.yaml"):
    """
    Displays a frame with a coordinate grid and coordinate annotations to help
    users select precise ROI and Stop Line coordinates for config.yaml.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not load frame {frame_number} from {video_path}")
        return

    h, w = frame.shape[:2]
    grid_frame = frame.copy()

    # Draw horizontal and vertical grid lines every 100 pixels
    grid_step = 100
    for x in range(0, w, grid_step):
        cv2.line(grid_frame, (x, 0), (x, h), (100, 100, 100), 1)
        cv2.putText(grid_frame, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for y in range(0, h, grid_step):
        cv2.line(grid_frame, (0, y), (w, y), (100, 100, 100), 1)
        cv2.putText(grid_frame, str(y), (5, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_image_path = f"calibration_grid_frame_{frame_number}.jpg"
    cv2.imwrite(out_image_path, grid_frame)
    print(f"Saved calibration grid frame to: {out_image_path}")
    print(f"Frame resolution: {w}x{h}")
    print("Inspect the image to update traffic_light_roi and stop_line in config.yaml")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "traffic_sample.mp4"
    run_calibration_tool(video)
