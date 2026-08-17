# -*- coding: utf-8 -*-

"""
Task B - Paragraph Extraction

BATCH VERSION

This program processes:
    001.png
    002.png
    003.png
    004.png
    005.png
    006.png
    007.png
    008.png

The images are read from the input folder.

The results are saved into:

    output/001
    output/002
    ...
    output/008

No if, elif, else, for, while or glob is used.
"""

import os
import cv2

import projection_v2


# ============================================================
# TASK B FOLDER
# ============================================================

task_b_folder = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FUNCTION: PROCESS ONE IMAGE
# ============================================================

def process_image(image_name):
    """
    Process one scientific paper image.

    image_name examples:
        "001"
        "002"
        ...
        "008"
    """

    print(
        "Processing:",
        image_name + ".png"
    )


    # ========================================================
    # CREATE INPUT PATH
    # ========================================================

    image_path = os.path.join(
        task_b_folder,
        "input",
        image_name + ".png"
    )


    # ========================================================
    # CREATE OUTPUT PATH
    # ========================================================

    output_folder = os.path.join(
        task_b_folder,
        "output",
        image_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )


    # ========================================================
    # CONVERT TO GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # FIND OTSU THRESHOLD
    # ========================================================

    otsu_value, otsu_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(
        image_name,
        "Otsu threshold:",
        otsu_value
    )


    # ========================================================
    # APPLY BINARY THRESHOLD
    # ========================================================

    # Text becomes black.
    # Background becomes white.
    threshold_value, binary = cv2.threshold(
        gray,
        otsu_value,
        255,
        cv2.THRESH_BINARY
    )


    # ========================================================
    # ORIGINAL HISTOGRAM PROJECTIONS
    # ========================================================

    horizontal = projection.horizontal_projection(
        binary
    )

    vertical = projection.vertical_projection(
        binary
    )


    # ========================================================
    # DETECT HORIZONTAL TABLE LINES
    # ========================================================

    horizontal_lines = projection.horizontal_line_detection(
        binary
    )


    # ========================================================
    # DETECT VERTICAL TABLE LINES
    # ========================================================

    vertical_lines = projection.vertical_line_detection(
        binary
    )


    # ========================================================
    # COMBINE TABLE LINES
    # ========================================================

    table_lines = projection.combine_table_lines(
        horizontal_lines,
        vertical_lines
    )


    # ========================================================
    # REMOVE TABLE LINES
    # ========================================================

    clean_binary = projection.remove_table_lines(
        binary,
        table_lines
    )


    # ========================================================
    # CLEAN HISTOGRAM PROJECTIONS
    # ========================================================

    clean_horizontal = projection.horizontal_projection(
        clean_binary
    )

    clean_vertical = projection.vertical_projection(
        clean_binary
    )


    # ========================================================
    # CREATE PARAGRAPH MASK
    # ========================================================

    paragraph_mask = projection.create_paragraph_mask(
        clean_binary
    )


    # ========================================================
    # SAVE BINARY IMAGE
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "binary.png"
        ),
        binary
    )


    # ========================================================
    # SAVE HORIZONTAL TABLE LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "horizontal_lines.png"
        ),
        horizontal_lines
    )


    # ========================================================
    # SAVE VERTICAL TABLE LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "vertical_lines.png"
        ),
        vertical_lines
    )


    # ========================================================
    # SAVE COMBINED TABLE LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "table_lines.png"
        ),
        table_lines
    )


    # ========================================================
    # SAVE CLEAN BINARY IMAGE
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "clean_binary.png"
        ),
        clean_binary
    )


    # ========================================================
    # SAVE PARAGRAPH MASK
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "paragraph_mask.png"
        ),
        paragraph_mask
    )


    # ========================================================
    # SAVE HORIZONTAL PROJECTION
    # ========================================================

    projection.save_projection(
        horizontal,
        image_name + " Horizontal Histogram Projection",
        "Image Row",
        os.path.join(
            output_folder,
            "horizontal_projection.png"
        )
    )


    # ========================================================
    # SAVE VERTICAL PROJECTION
    # ========================================================

    projection.save_projection(
        vertical,
        image_name + " Vertical Histogram Projection",
        "Image Column",
        os.path.join(
            output_folder,
            "vertical_projection.png"
        )
    )


    # ========================================================
    # SAVE CLEAN HORIZONTAL PROJECTION
    # ========================================================

    projection.save_projection(
        clean_horizontal,
        image_name + " Horizontal Projection After Table Removal",
        "Image Row",
        os.path.join(
            output_folder,
            "clean_horizontal_projection.png"
        )
    )


    # ========================================================
    # SAVE CLEAN VERTICAL PROJECTION
    # ========================================================

    projection.save_projection(
        clean_vertical,
        image_name + " Vertical Projection After Table Removal",
        "Image Column",
        os.path.join(
            output_folder,
            "clean_vertical_projection.png"
        )
    )


    print(
        image_name,
        "complete."
    )


# ============================================================
# PROCESS 001
# ============================================================

process_image(
    "001"
)


# ============================================================
# PROCESS 002
# ============================================================

process_image(
    "002"
)


# ============================================================
# PROCESS 003
# ============================================================

process_image(
    "003"
)


# ============================================================
# PROCESS 004
# ============================================================

process_image(
    "004"
)


# ============================================================
# PROCESS 005
# ============================================================

process_image(
    "005"
)


# ============================================================
# PROCESS 006
# ============================================================

process_image(
    "006"
)


# ============================================================
# PROCESS 007
# ============================================================

process_image(
    "007"
)


# ============================================================
# PROCESS 008
# ============================================================

process_image(
    "008"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "All Task B images have been processed."
)