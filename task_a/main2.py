import os
import cv2
from brightnessJordan import detect_brightness, adjust_brightness
from face_blur import blur_faces

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_video(input_path, output_path):
    print(f"Processing {input_path}...")
    print(f"Output will be saved to {output_path}")
    print(f"Editing Video...")
    night = detect_brightness(input_path)


    #Get video properties for output
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    #Create output video writer
    output = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        fps,
                        (width, height))
    
    print(f"Video properties: fps={fps}, width={width}, height={height}, total_frames={total_frames}")
    frame_count = 0

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

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename == "talking.mp4":
            continue
        input_path = os.path.join(INPUT_DIR, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed.avi")
        process_video(input_path, output_path)
    

    



