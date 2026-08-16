import numpy as np
import cv2
from matplotlib import pyplot as plt


##function to show image (so code is more efficient a bit)
def show_image(image, title, color_map=None):
    plt.figure(figsize=(10, 12))
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#Reading image
image = cv2.imread("task_b\input\008.png")

#Converting image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGRGRAY)

#Finding Otsu threshold then displaying it
otsu_value, _ = cv2.threshold(gray, 0, 225, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("The otsu threshold value is : ", otsu_value)

#Applying Binary Threshold
threshold_value, binary = cv2.threshold(gray, otsu_value, 255, cv2.THRESH_BINARY_INV)

#Creating horizontal + vertical projection
horizontal_projection = np.count_nonzero(binary, axis=1)
vertical_projection = np.count_nonzero(binary, axis=0)

#Detecting horizontal table lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(80, 1))