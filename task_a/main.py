import os
import cv2
from brightness import is_nighttime, adjust_brightness

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def process_video(input_path, output_path):
    night, avg_brightness = is_nighttime(input_path)
    print(f"{input_path}: avg_brightness={avg_brightness:.2f}, night={night}")

    vid = cv2.VideoCapture(input_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )

    for _ in range(total_frames):
        success, frame = vid.read()
        if not success:
            break
        if night:
            frame = adjust_brightness(frame)
        out.write(frame)

    vid.release()
    out.release()


if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename == "talking.mp4":
            continue
        input_path = os.path.join(INPUT_DIR, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed.avi")
        process_video(input_path, output_path)
