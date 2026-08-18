import cv2

def append_endscreen(output_writer, endscreen_path):
    #open endscreen video
    endscreen_vid = cv2.VideoCapture(endscreen_path)

    while True:
        #read the next frame from the endscreen video
        success, end_frame = endscreen_vid.read()
        #stop when no next more frames
        if not success:
            break
        #write the frame into main video
        output_writer.write(end_frame)
    