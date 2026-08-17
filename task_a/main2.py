import os
import cv2
from brightnessJordan import detect_brightness, adjust_brightness
from face_blur import blur_faces

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_video(input_path, output_path):
    night = detect_brightness(input_path)

    #Get video properties for output
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    #Create output video writer
    output = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        fps,
                        (width, height))
        
    #Open the video file and get number of frames
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0, total_frames):
        #Jump to the specific frame and read it
        video.set(cv2.CAP_PROP_POS_FRAMES, frame) 
        sucess, image = video.read()
        if not sucess:
            print(f"Failed to read frame {frame} from {input_path}")
            break

        #If night time, adjust brightness
        if night:
            image = adjust_brightness(image)

        image = blur_faces(image)

        output.write(image)


if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename == "talking.mp4":
            continue
        input_path = os.path.join(INPUT_DIR, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed.avi")
        process_video(input_path, output_path)
    

    



