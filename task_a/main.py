import os
import cv2
from brightness import detect_brightness, adjust_brightness
from face_blur import blur_faces
from overlay import overlay_talking,add_watermark

#INPUT DIRECTORY NAME
INPUT_DIR = "input"
#OUTPUT DIRECTORY NAME
OUTPUT_DIR = "output"

#Function to start video processing
def process_video(input_path, output_path):
    print(f"Processing {input_path}...")
    print(f"Output will be saved to {output_path}")
    print(f"Editing Video...")
    #call detect brightness to determine time of day
    night = detect_brightness(input_path)


    #Read the Video
    video = cv2.VideoCapture(input_path)
    #Get Video FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    #Get Video Width
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #Get Video Height
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    #Create output video writer
    output = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        fps,
                        (width, height))
    
    print(f"Video properties: fps={fps}, width={width}, height={height}, total_frames={total_frames}")

    #Read video of the girl talking
    talking_vid = cv2.VideoCapture("./input/talking.mp4")
    #Initalize the frame counter variable
    frame_count = 0

    #iterate over each frame and perform processing on each frame
    for frame in range(0, total_frames):
        #Read the current frame
        sucess, image = video.read()
        #If we fail to read the file, throw error
        if not sucess:
            print(f"Failed to read frame {frame} from {input_path}")
            break

        #If night time, adjust brightness
        if night:
            image = adjust_brightness(image)

        #blur faces
        image = blur_faces(image)

        #alternate between watermark every 10 seconds
        watermark1 = cv2.imread('./assets/watermark1.png')
        watermark2 = cv2.imread('./assets/watermark2.png')
        block = int(fps * 10)

        #if it is an even 10 seconds block use watermark1
        #else use watermark2
        if (frame_count // block) % 2 == 0:
            watermark = watermark1
        else:
            watermark = watermark2
            
        #call add_watermark function with x,y position of 0,0 
        #x and y is the location to put the watermark on the frame
        image = add_watermark(image, watermark, 0, 0, 0.2)

        #adding overlay
        success_talk, talk_frame = talking_vid.read()
        #check if the duration of the talking video is shorter than the video being processed
        if success_talk:
            image = overlay_talking(image,talk_frame)
        
        #write the processed frame to the output video
        output.write(image)

        #increment the frame counter
        frame_count += 1
        #display number of frames processed every 100 frames 
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

if __name__ == "__main__":
    #Iterate over each file in inputs directory
    for filename in os.listdir(INPUT_DIR):
        #prepend the filename with the input path
        input_path = os.path.join(INPUT_DIR, filename)
        #extract the filename from the input file
        name, _ = os.path.splitext(filename)
        #prepend the filename with the output path
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed.avi")
        process_video(input_path, output_path)
    

    



