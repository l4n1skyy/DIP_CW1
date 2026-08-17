import numpy as np
import cv2
from matplotlib import pyplot as plt

import projection #this imports the projection.py file with functions on the same folder

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


### Histogram projection section ###


#counts the number of black text pixels in every image row and column
horizontal = projection.horizontal_projection(binary)
vertical = projection.vertical_projection(binary)


#finding the starting and ending row text ranges
row_start, row_end = projection.find_projection_ranges(horizontal)
#finding the starting and ending column text ranges
column_start, column_end = projection.find_projection_ranges(vertical)


#making a inverted binary image for morphological processing
binary_inverse = cv2.bitwise_not(binary)


### Detection segment ###

# Create a horizontal structuring element.
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (80, 1)
)

# Detect long horizontal lines.
horizontal_lines = cv2.morphologyEx(
    binary_inverse,
    cv2.MORPH_OPEN,
    horizontal_kernel
)


# ============================================================
# STEP 11: DETECT VERTICAL TABLE LINES
# ============================================================

# Create a vertical structuring element.
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, 40)
)

# Detect long vertical lines.
vertical_lines = cv2.morphologyEx(
    binary_inverse,
    cv2.MORPH_OPEN,
    vertical_kernel
)


# ============================================================
# STEP 12: COMBINE THE TABLE LINES
# ============================================================

# Combine the horizontal and vertical table lines.
table_lines = cv2.bitwise_or(
    horizontal_lines,
    vertical_lines
)


# ============================================================
# STEP 13: FIND THE TABLE REGION
# ============================================================

# Find the outer boundary of the table.
table_contours, hierarchy = cv2.findContours(
    table_lines,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ============================================================
# STEP 14: CREATE THE TABLE MASK
# ============================================================

# Create an empty image.
table_mask = np.zeros_like(
    binary
)

# Fill the detected table region.
cv2.drawContours(
    table_mask,
    table_contours,
    -1,
    255,
    cv2.FILLED
)


# ============================================================
# STEP 15: REMOVE THE TABLE
# ============================================================

# Invert the table mask.
inverse_table_mask = cv2.bitwise_not(
    table_mask
)

# Remove the table from the inverted binary image.
text_only_inverse = cv2.bitwise_and(
    binary_inverse,
    inverse_table_mask
)


# ============================================================
# STEP 16: CREATE A CLEAN BINARY IMAGE
# ============================================================

# Convert the cleaned image back to black text.
clean_binary = cv2.bitwise_not(
    text_only_inverse
)


# ============================================================
# STEP 17: CREATE A CLEAN VERTICAL PROJECTION
# ============================================================

# Calculate the vertical projection after table removal.
clean_vertical = projection.vertical_projection(
    clean_binary
)


# ============================================================
# STEP 18: FIND THE GAP BETWEEN THE TWO COLUMNS
# ============================================================

# Find the position of the gap between the text columns.
column_split = projection.find_column_split(
    clean_vertical
)

print(
    "Column split position:",
    column_split
)


# ============================================================
# STEP 19: CONNECT TEXT INTO PARAGRAPH REGIONS
# ============================================================

# Create a rectangular structuring element.
paragraph_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (25, 20)
)

# Connect nearby letters, words and text lines.
paragraph_mask = cv2.dilate(
    text_only_inverse,
    paragraph_kernel,
    iterations=1
)


# ============================================================
# STEP 20: FIND CONNECTED PARAGRAPH REGIONS
# ============================================================

# Find all connected components.
number_labels, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(
        paragraph_mask,
        connectivity=8
    )

# Remove the background component.
stats = stats[
    1:
]


# ============================================================
# STEP 21: REMOVE SMALL REGIONS
# ============================================================

# Keep regions that are large enough to represent paragraphs.
valid = (
    (
        stats[
            :,
            cv2.CC_STAT_WIDTH
        ] > 250
    )
    &
    (
        stats[
            :,
            cv2.CC_STAT_HEIGHT
        ] > 50
    )
    &
    (
        stats[
            :,
            cv2.CC_STAT_AREA
        ] > 10000
    )
)

# Keep the valid paragraph regions.
paragraph_stats = stats[
    valid
]


# ============================================================
# STEP 22: SORT THE PARAGRAPHS
# ============================================================

# Use the projection module to arrange the paragraphs.
sorted_stats = projection.sort_paragraphs(
    paragraph_stats,
    column_split
)


# ============================================================
# STEP 23: GET THE DETECTED REGIONS
# ============================================================

# Paragraph 1.
x1, y1, w1, h1, a1 = sorted_stats[0]

# Paragraph 2.
x2, y2, w2, h2, a2 = sorted_stats[1]

# First part of Paragraph 3.
x3a, y3a, w3a, h3a, a3a = sorted_stats[2]

# Continuation of Paragraph 3.
x3b, y3b, w3b, h3b, a3b = sorted_stats[3]

# Paragraph 4.
x4, y4, w4, h4, a4 = sorted_stats[4]

# Paragraph 5.
x5, y5, w5, h5, a5 = sorted_stats[5]

# Paragraph 6.
x6, y6, w6, h6, a6 = sorted_stats[6]


# ============================================================
# STEP 24: EXTRACT PARAGRAPH 1
# ============================================================

paragraph1 = original[
    y1:y1+h1,
    x1:x1+w1
]


# ============================================================
# STEP 25: EXTRACT PARAGRAPH 2
# ============================================================

paragraph2 = original[
    y2:y2+h2,
    x2:x2+w2
]


# ============================================================
# STEP 26: EXTRACT PARAGRAPH 3 - FIRST PART
# ============================================================

paragraph3_left = original[
    y3a:y3a+h3a,
    x3a:x3a+w3a
]


# ============================================================
# STEP 27: EXTRACT PARAGRAPH 3 - CONTINUATION
# ============================================================

paragraph3_right = original[
    y3b:y3b+h3b,
    x3b:x3b+w3b
]


# ============================================================
# STEP 28: EXTRACT PARAGRAPH 4
# ============================================================

paragraph4 = original[
    y4:y4+h4,
    x4:x4+w4
]


# ============================================================
# STEP 29: EXTRACT PARAGRAPH 5
# ============================================================

paragraph5 = original[
    y5:y5+h5,
    x5:x5+w5
]


# ============================================================
# STEP 30: EXTRACT PARAGRAPH 6
# ============================================================

paragraph6 = original[
    y6:y6+h6,
    x6:x6+w6
]


# ============================================================
# STEP 31: JOIN THE TWO PARTS OF PARAGRAPH 3
# ============================================================

# Set a small white space between both parts.
gap = 20

# Find the largest width.
paragraph3_width = int(
    np.maximum(
        paragraph3_left.shape[1],
        paragraph3_right.shape[1]
    )
)

# Calculate the total height.
paragraph3_height = (
    paragraph3_left.shape[0]
    +
    paragraph3_right.shape[0]
    +
    gap
)

# Create a white image for the complete paragraph.
paragraph3 = np.full(
    (
        paragraph3_height,
        paragraph3_width,
        3
    ),
    255,
    dtype=np.uint8
)

# Place the first part at the top.
paragraph3[
    0:paragraph3_left.shape[0],
    0:paragraph3_left.shape[1]
] = paragraph3_left

# Calculate where the continuation starts.
paragraph3_start = (
    paragraph3_left.shape[0]
    +
    gap
)

# Place the continuation below the first part.
paragraph3[
    paragraph3_start:
    paragraph3_start + paragraph3_right.shape[0],
    0:paragraph3_right.shape[1]
] = paragraph3_right


# ============================================================
# STEP 32: DRAW THE DETECTED PARAGRAPHS
# ============================================================

# Create a copy for the final detection image.
detected = original.copy()

# Draw Paragraph 1.
cv2.rectangle(
    detected,
    (x1, y1),
    (x1+w1, y1+h1),
    (0, 0, 255),
    3
)

# Draw Paragraph 2.
cv2.rectangle(
    detected,
    (x2, y2),
    (x2+w2, y2+h2),
    (0, 0, 255),
    3
)

# Draw the first part of Paragraph 3.
cv2.rectangle(
    detected,
    (x3a, y3a),
    (x3a+w3a, y3a+h3a),
    (0, 0, 255),
    3
)

# Draw the continuation of Paragraph 3.
cv2.rectangle(
    detected,
    (x3b, y3b),
    (x3b+w3b, y3b+h3b),
    (0, 0, 255),
    3
)

# Draw Paragraph 4.
cv2.rectangle(
    detected,
    (x4, y4),
    (x4+w4, y4+h4),
    (0, 0, 255),
    3
)

# Draw Paragraph 5.
cv2.rectangle(
    detected,
    (x5, y5),
    (x5+w5, y5+h5),
    (0, 0, 255),
    3
)

# Draw Paragraph 6.
cv2.rectangle(
    detected,
    (x6, y6),
    (x6+w6, y6+h6),
    (0, 0, 255),
    3
)


# ============================================================
# STEP 33: SAVE THE EXTRACTED PARAGRAPHS
# ============================================================

cv2.imwrite(
    "paragraph_1.png",
    paragraph1
)

cv2.imwrite(
    "paragraph_2.png",
    paragraph2
)

cv2.imwrite(
    "paragraph_3.png",
    paragraph3
)

cv2.imwrite(
    "paragraph_4.png",
    paragraph4
)

cv2.imwrite(
    "paragraph_5.png",
    paragraph5
)

cv2.imwrite(
    "paragraph_6.png",
    paragraph6
)


# ============================================================
# STEP 34: SAVE THE PROCESSING OUTPUTS
# ============================================================

cv2.imwrite(
    "binary.png",
    binary
)

cv2.imwrite(
    "table_lines.png",
    table_lines
)

cv2.imwrite(
    "table_mask.png",
    table_mask
)

cv2.imwrite(
    "clean_binary.png",
    clean_binary
)

cv2.imwrite(
    "paragraph_mask.png",
    paragraph_mask
)

cv2.imwrite(
    "detected_paragraphs.png",
    detected
)


# ============================================================
# STEP 35: DISPLAY THE ORIGINAL IMAGE
# ============================================================

show_image(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    ),
    "Original Paper"
)


# ============================================================
# STEP 36: DISPLAY THE GRAYSCALE IMAGE
# ============================================================

show_image(
    gray,
    "Grayscale Image",
    "gray"
)


# ============================================================
# STEP 37: DISPLAY THE BINARY IMAGE
# ============================================================

show_image(
    binary,
    "Binary Image",
    "gray"
)


# ============================================================
# STEP 38: DISPLAY THE PARAGRAPH MASK
# ============================================================

show_image(
    paragraph_mask,
    "Paragraph Regions",
    "gray"
)


# ============================================================
# STEP 39: DISPLAY THE FINAL RESULT
# ============================================================

show_image(
    cv2.cvtColor(
        detected,
        cv2.COLOR_BGR2RGB
    ),
    "Detected Paragraphs"
)


# ============================================================
# STEP 40: DISPLAY THE HORIZONTAL PROJECTION
# ============================================================

projection.show_projection(
    horizontal,
    "Horizontal Histogram Projection",
    "Image Row"
)


# ============================================================
# STEP 41: DISPLAY THE VERTICAL PROJECTION
# ============================================================

projection.show_projection(
    vertical,
    "Vertical Histogram Projection",
    "Image Column"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "Number of detected text regions:",
    len(sorted_stats)
)

print(
    "Number of extracted paragraphs: 6"
)

print(
    "Task B is complete."