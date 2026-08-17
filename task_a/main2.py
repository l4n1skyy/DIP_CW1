import os
import cv2
from brightness import detect_brightness, adjust_brightness
from face_blur import blur_faces
from overlay import overlay_talking,add_watermark

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

    talking_vid = cv2.VideoCapture("./input/talking.mp4")

    for frame in range(0, total_frames):
        #Jump to the specific frame and read it
        video.set(cv2.CAP_PROP_POS_FRAMES, frame) 
        sucess, image = video.read()
        if not sucess:
            print(f"Failed to read frame {frame} from {input_path}")
            break
        # talking_vid.set(cv2.CAP_PROP_POS_FRAMES,frame)

        #If night time, adjust brightness
        if night:
            image = adjust_brightness(image)

        #blur faces
        image = blur_faces(image)

        #alternating water mark based on frame 
        watermark1 = cv2.imread('./assets/watermark1.png')
        watermark2 = cv2.imread('./assets/watermark2.png')
        block = int(fps * 10)

        if (frame_count // block) % 2 == 0:
            watermark = watermark1
        else:
            watermark = watermark2

        image = add_watermark(image, watermark, 0, 0, 0.2)

        #adding overlay
        success_talk, talk_frame = talking_vid.read()
        if success_talk:
            image = overlay_talking(image,talk_frame)
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
    

    



