# -*- coding: utf-8 -*-

"""
Task B - Paragraph Extraction

SINGLE IMAGE VERSION

This program processes one scientific paper image
at a time.

Only change the image_name before each run.

Example:

    image_name = "001"
    image_name = "002"
    ...
    image_name = "008"

No if, elif, else, for, while or glob is used.
"""

import os
import cv2

import projection_v2


# ============================================================
# STEP 1: SELECT IMAGE
# ============================================================

# Change this value to select another paper.
image_name = "008"


# ============================================================
# STEP 2: CREATE INPUT AND OUTPUT PATHS
# ============================================================

# Find the folder containing this main file.
task_b_folder = os.path.dirname(
    os.path.abspath(__file__)
)

# Create the input image path.
image_path = os.path.join(
    task_b_folder,
    "input",
    image_name + ".png"
)

# Create the output folder path.
output_folder = os.path.join(
    task_b_folder,
    "output",
    image_name
)

# Create the output folder when required.
os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# STEP 3: READ IMAGE
# ============================================================

image = cv2.imread(
    image_path
)


# ============================================================
# STEP 4: CONVERT TO GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# STEP 5: FIND OTSU THRESHOLD
# ============================================================

# Otsu automatically finds the threshold value.
otsu_value, otsu_image = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(
    "Otsu threshold value:",
    otsu_value
)


# ============================================================
# STEP 6: APPLY BINARY THRESHOLD
# ============================================================

# Text becomes black.
# Background becomes white.
threshold_value, binary = cv2.threshold(
    gray,
    otsu_value,
    255,
    cv2.THRESH_BINARY
)


# ============================================================
# STEP 7: ORIGINAL HORIZONTAL PROJECTION
# ============================================================

horizontal = projection.horizontal_projection(
    binary
)


# ============================================================
# STEP 8: ORIGINAL VERTICAL PROJECTION
# ============================================================

vertical = projection.vertical_projection(
    binary
)


# ============================================================
# STEP 9: DETECT HORIZONTAL TABLE LINES
# ============================================================

horizontal_lines = projection.horizontal_line_detection(
    binary
)


# ============================================================
# STEP 10: DETECT VERTICAL TABLE LINES
# ============================================================

vertical_lines = projection.vertical_line_detection(
    binary
)


# ============================================================
# STEP 11: COMBINE TABLE LINES
# ============================================================

table_lines = projection.combine_table_lines(
    horizontal_lines,
    vertical_lines
)


# ============================================================
# STEP 12: REMOVE TABLE LINES
# ============================================================

clean_binary = projection.remove_table_lines(
    binary,
    table_lines
)


# ============================================================
# STEP 13: CLEAN HORIZONTAL PROJECTION
# ============================================================

clean_horizontal = projection.horizontal_projection(
    clean_binary
)


# ============================================================
# STEP 14: CLEAN VERTICAL PROJECTION
# ============================================================

clean_vertical = projection.vertical_projection(
    clean_binary
)


# ============================================================
# STEP 15: CREATE PARAGRAPH MASK
# ============================================================

paragraph_mask = projection.create_paragraph_mask(
    clean_binary
)


# ============================================================
# STEP 16: SAVE BINARY IMAGE
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "binary.png"
    ),
    binary
)


# ============================================================
# STEP 17: SAVE HORIZONTAL TABLE LINES
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "horizontal_lines.png"
    ),
    horizontal_lines
)


# ============================================================
# STEP 18: SAVE VERTICAL TABLE LINES
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "vertical_lines.png"
    ),
    vertical_lines
)


# ============================================================
# STEP 19: SAVE COMBINED TABLE LINES
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "table_lines.png"
    ),
    table_lines
)


# ============================================================
# STEP 20: SAVE CLEAN BINARY IMAGE
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "clean_binary.png"
    ),
    clean_binary
)


# ============================================================
# STEP 21: SAVE PARAGRAPH MASK
# ============================================================

cv2.imwrite(
    os.path.join(
        output_folder,
        "paragraph_mask.png"
    ),
    paragraph_mask
)


# ============================================================
# STEP 22: SAVE HORIZONTAL PROJECTION
# ============================================================

projection.save_projection(
    horizontal,
    "Horizontal Histogram Projection - " + image_name,
    "Image Row",
    os.path.join(
        output_folder,
        "horizontal_projection.png"
    )
)


# ============================================================
# STEP 23: SAVE VERTICAL PROJECTION
# ============================================================

projection.save_projection(
    vertical,
    "Vertical Histogram Projection - " + image_name,
    "Image Column",
    os.path.join(
        output_folder,
        "vertical_projection.png"
    )
)


# ============================================================
# STEP 24: SAVE CLEAN HORIZONTAL PROJECTION
# ============================================================

projection.save_projection(
    clean_horizontal,
    "Horizontal Projection After Table Removal - " + image_name,
    "Image Row",
    os.path.join(
        output_folder,
        "clean_horizontal_projection.png"
    )
)


# ============================================================
# STEP 25: SAVE CLEAN VERTICAL PROJECTION
# ============================================================

projection.save_projection(
    clean_vertical,
    "Vertical Projection After Table Removal - " + image_name,
    "Image Column",
    os.path.join(
        output_folder,
        "clean_vertical_projection.png"
    )
)


# ============================================================
# STEP 26: DISPLAY ORIGINAL IMAGE
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

projection.show_image(
    image_rgb,
    "Original Paper - " + image_name
)


# ============================================================
# STEP 27: DISPLAY GRAYSCALE IMAGE
# ============================================================

projection.show_image(
    gray,
    "Grayscale Image - " + image_name,
    "gray"
)


# ============================================================
# STEP 28: DISPLAY BINARY IMAGE
# ============================================================

projection.show_image(
    binary,
    "Binary Image - " + image_name,
    "gray"
)


# ============================================================
# STEP 29: DISPLAY TABLE LINES
# ============================================================

projection.show_image(
    table_lines,
    "Detected Table Lines - " + image_name,
    "gray"
)


# ============================================================
# STEP 30: DISPLAY CLEAN BINARY IMAGE
# ============================================================

projection.show_image(
    clean_binary,
    "Image After Table Line Removal - " + image_name,
    "gray"
)


# ============================================================
# STEP 31: DISPLAY PARAGRAPH MASK
# ============================================================

projection.show_image(
    paragraph_mask,
    "Paragraph Mask - " + image_name,
    "gray"
)


# ============================================================
# STEP 32: DISPLAY HORIZONTAL PROJECTION
# ============================================================

projection.show_projection(
    horizontal,
    "Horizontal Histogram Projection - " + image_name,
    "Image Row"
)


# ============================================================
# STEP 33: DISPLAY VERTICAL PROJECTION
# ============================================================

projection.show_projection(
    vertical,
    "Vertical Histogram Projection - " + image_name,
    "Image Column"
)


# ============================================================
# STEP 34: DISPLAY CLEAN HORIZONTAL PROJECTION
# ============================================================

projection.show_projection(
    clean_horizontal,
    "Horizontal Projection After Table Removal - " + image_name,
    "Image Row"
)


# ============================================================
# STEP 35: DISPLAY CLEAN VERTICAL PROJECTION
# ============================================================

projection.show_projection(
    clean_vertical,
    "Vertical Projection After Table Removal - " + image_name,
    "Image Column"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "Image processed:",
    image_name + ".png"
)

print(
    "Output folder:",
    output_folder
)

print(
    "Processing is complete."
)