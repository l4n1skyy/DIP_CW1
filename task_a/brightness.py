import cv2
import numpy as np


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def is_nighttime(video_path, sample_count=10, threshold=90):
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // sample_count, 1)

    brightness_values = []
    for frame_index in range(0, total_frames, step):
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = vid.read()
        if success:
            brightness_values.append(get_brightness(frame))

    vid.release()
    avg_brightness = np.mean(brightness_values)
    return avg_brightness < threshold, avg_brightness


def adjust_brightness(frame, increase_by=60):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + increase_by, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
