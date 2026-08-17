import cv2
import numpy as np

def detect_brightness(video_path):
    total_dark_pixels = 0
    total_pixels = 0

    # Open the video file and get number of frames
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0, total_frames, 30):
        # Jump to the specific frame and read it
        video.set(cv2.CAP_PROP_POS_FRAMES, frame) 
        sucess, image = video.read()

        if sucess:
            # Convert the frame to grayscale and histogram
            gray_scale_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray_scale_frame], [0], None, [256], [0, 256])

            # Count pixels in dark range (0-80) and total pixels
            dark_pixels = sum(histogram[i][0] for i in range(0, 81))
            pixels = gray_scale_frame.shape[0] * gray_scale_frame.shape[1]

            total_dark_pixels += dark_pixels
            total_pixels += pixels
    
    darkness_ratio = total_dark_pixels/total_pixels

    #Returns true if video is night
    if darkness_ratio > 0.6:
        print("Video is night")
        return True
    else:
        print("Video is day")
        return False

def adjust_brightness(frame):
        #Add brightness
        image_float = frame.astype(np.float64)
        brightened_image = image_float + 50
        brightened_image = np.clip(brightened_image, 0, 255)
        brightened_image = brightened_image.astype(np.uint8)
        return brightened_image



