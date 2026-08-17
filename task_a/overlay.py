import cv2

def overlay_talking(frame, talk_frame, pip_w=320, pip_h=180):
    frame[0:pip_h, 0:pip_w] = cv2.resize(talk_frame, (pip_w, pip_h))
    cv2.rectangle(frame, (0, 0), (pip_w - 1, pip_h - 1), (0, 0, 0), 3)
    return frame


def add_watermark(frame, watermark, x, y, alpha=0.5):
    h, w = watermark.shape[:2]
    roi = frame[y : y + h, x : x + w]

    if watermark.shape[2] == 4:
        wm_rgb = watermark[:, :, :3].astype(float)
        wm_alpha = (watermark[:, :, 3:4].astype(float) / 255.0) * alpha
        blended = wm_alpha * wm_rgb + (1 - wm_alpha) * roi.astype(float)
        frame[y : y + h, x : x + w] = blended.astype("uint8")
    else:
        frame[y : y + h, x : x + w] = cv2.addWeighted(
            watermark, alpha, roi, 1 - alpha, 0
        )

    return frame
