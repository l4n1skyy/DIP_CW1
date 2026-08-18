import cv2

def overlay_talking(frame, talk_frame, pip_w=320, pip_h=180):
    #resize the talking video to the width and height defined
    #default is 320x180
    resized_talking = cv2.resize(talk_frame, (pip_w, pip_h))
    #replace the original pixels in the current frame with the talking video's pixels
    frame[0:pip_h, 0:pip_w] = resized_talking
    #add a black(0,0,0) border around the talking video
    cv2.rectangle(frame, (0, 0), (pip_w - 1, pip_h - 1), (0, 0, 0), 3)
    return frame


def add_watermark(frame, watermark, x, y, alpha=0.5):
    #get the height and width of the watermark
    h, w = watermark.shape[:2]
    #define the region of interest in the current frame
    #region of interest = the are we will be putting our watermark
    roi = frame[y : y + h, x : x + w]
    #blend the original pixels with the watermark pixels 
    frame[y : y + h, x : x + w] = cv2.addWeighted(
        watermark, alpha, roi, 1 - alpha, 0
    )

    return frame
