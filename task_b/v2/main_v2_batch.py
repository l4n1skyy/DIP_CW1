# -*- coding: utf-8 -*-

"""
Task B - Paragraph Extraction

Batch Processing Version

The program processes:

    001.png
    002.png
    003.png
    004.png
    005.png
    006.png
    007.png
    008.png

Input images are stored inside:

    input/

Results are stored inside:

    output/001/
    output/002/
    ...
    output/008/
"""

import os
import cv2
import numpy as np
import projection_v2


# ============================================================
# TASK B FOLDER
# ============================================================

# Get the folder where main.py is located.
task_b_folder = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FUNCTION: PROCESS ONE PAPER IMAGE
# ============================================================

def process_image(image_name):
    """
    Process one scientific paper image.

    Example:

        image_name = "001"

    reads:

        input/001.png

    and saves results into:

        output/001/
    """

    print(
        "\nProcessing:",
        image_name + ".png"
    )


    # ========================================================
    # STEP 1: CREATE INPUT IMAGE PATH
    # ========================================================

    image_path = os.path.join(
        task_b_folder,
        "input",
        image_name + ".png"
    )


    # ========================================================
    # STEP 2: CREATE OUTPUT FOLDER PATH
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
    # STEP 3: READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    # Keep original image for paragraph cropping.
    original = image.copy()


    # ========================================================
    # STEP 4: CONVERT TO GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # STEP 5: FIND OTSU THRESHOLD
    # ========================================================

    # Otsu automatically finds the threshold value.
    otsu_value, otsu_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    print(
        "Otsu threshold:",
        otsu_value
    )


    # ========================================================
    # STEP 6: APPLY BINARY THRESHOLD
    # ========================================================

    # Text becomes black = 0.
    # Background becomes white = 255.
    threshold_value, binary = cv2.threshold(
        gray,
        otsu_value,
        255,
        cv2.THRESH_BINARY
    )


    # ========================================================
    # STEP 7: HORIZONTAL HISTOGRAM PROJECTION
    # ========================================================

    horizontal = projection_v2.horizontal_projection(
        binary
    )


    # ========================================================
    # STEP 8: VERTICAL HISTOGRAM PROJECTION
    # ========================================================

    vertical = projection_v2.vertical_projection(
        binary
    )


    # ========================================================
    # STEP 9: DETECT HORIZONTAL TABLE LINES
    # ========================================================

    horizontal_lines = (
        projection_v2.horizontal_line_detection(
            binary
        )
    )


    # ========================================================
    # STEP 10: DETECT VERTICAL TABLE LINES
    # ========================================================

    vertical_lines = (
        projection_v2.vertical_line_detection(
            binary
        )
    )


    # ========================================================
    # STEP 11: COMBINE TABLE LINES
    # ========================================================

    table_lines = projection_v2.combine_table_lines(
        horizontal_lines,
        vertical_lines
    )


    # ========================================================
    # STEP 12: REMOVE TABLE LINES
    # ========================================================

    clean_binary = projection_v2.remove_table_lines(
        binary,
        table_lines
    )


    # ========================================================
    # STEP 13: CLEAN HORIZONTAL PROJECTION
    # ========================================================

    clean_horizontal = projection_v2.horizontal_projection(
        clean_binary
    )


    # ========================================================
    # STEP 14: CLEAN VERTICAL PROJECTION
    # ========================================================

    clean_vertical = projection_v2.vertical_projection(
        clean_binary
    )


    # ========================================================
    # STEP 15: CREATE PARAGRAPH MASK
    # ========================================================

    paragraph_mask = projection_v2.create_paragraph_mask(
        clean_binary
    )


    # ========================================================
    # STEP 16: FIND CONNECTED PARAGRAPH REGIONS
    # ========================================================

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            paragraph_mask,
            connectivity=8
        )
    )


    # ========================================================
    # STEP 17: REMOVE BACKGROUND REGION
    # ========================================================

    # Label 0 represents the image background.
    stats = stats[
        1:
    ]


    # ========================================================
    # STEP 18: REMOVE SMALL / NOISE REGIONS
    # ========================================================

    valid_regions = (

        (stats[:, cv2.CC_STAT_WIDTH] > 250)

        &

        (stats[:, cv2.CC_STAT_HEIGHT] > 50)

        &

        (stats[:, cv2.CC_STAT_AREA] > 10000)

    )

    paragraph_stats = stats[
        valid_regions
    ]

#bug fix for color images
    paragraph_stats = projection_v2.remove_colour_regions(
    original,
    paragraph_stats,
    saturation_threshold=15
)


    
    # ========================================================
    # STEP 19: SORT PARAGRAPHS
    # ========================================================

    sorted_stats = projection_v2.sort_paragraphs(
        paragraph_stats
    )


    # ========================================================
    # STEP 20: PREPARE DETECTION IMAGE
    # ========================================================

    detected = original.copy()


    # ========================================================
    # STEP 21: EXTRACT AND SAVE PARAGRAPHS
    # ========================================================

    for paragraph_number, paragraph in enumerate(
        sorted_stats,
        start=1
    ):


        # ----------------------------------------------------
        # Get paragraph position.
        # ----------------------------------------------------

        x = paragraph[
            cv2.CC_STAT_LEFT
        ]

        y = paragraph[
            cv2.CC_STAT_TOP
        ]


        # ----------------------------------------------------
        # Get paragraph size.
        # ----------------------------------------------------

        width = paragraph[
            cv2.CC_STAT_WIDTH
        ]

        height = paragraph[
            cv2.CC_STAT_HEIGHT
        ]


        # ----------------------------------------------------
        # Crop paragraph from original paper.
        # ----------------------------------------------------

        paragraph_image = original[
            y:y + height,
            x:x + width
        ]


        # ----------------------------------------------------
        # Create paragraph filename.
        # ----------------------------------------------------

        paragraph_path = os.path.join(
            output_folder,
            "paragraph_"
            + str(paragraph_number)
            + ".png"
        )


        # ----------------------------------------------------
        # Save paragraph.
        # ----------------------------------------------------

        cv2.imwrite(
            paragraph_path,
            paragraph_image
        )


        # ----------------------------------------------------
        # Draw bounding rectangle.
        # ----------------------------------------------------

        cv2.rectangle(
            detected,
            (
                x,
                y
            ),
            (
                x + width,
                y + height
            ),
            (
                0,
                0,
                255
            ),
            3
        )


    # ========================================================
    # STEP 22: SAVE BINARY IMAGE
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "binary.png"
        ),
        binary
    )


    # ========================================================
    # STEP 23: SAVE HORIZONTAL LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "horizontal_lines.png"
        ),
        horizontal_lines
    )


    # ========================================================
    # STEP 24: SAVE VERTICAL LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "vertical_lines.png"
        ),
        vertical_lines
    )


    # ========================================================
    # STEP 25: SAVE TABLE LINES
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "table_lines.png"
        ),
        table_lines
    )


    # ========================================================
    # STEP 26: SAVE CLEAN BINARY IMAGE
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "clean_binary.png"
        ),
        clean_binary
    )


    # ========================================================
    # STEP 27: SAVE PARAGRAPH MASK
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "paragraph_mask.png"
        ),
        paragraph_mask
    )


    # ========================================================
    # STEP 28: SAVE DETECTED PARAGRAPHS
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "detected_paragraphs.png"
        ),
        detected
    )


    # ========================================================
    # STEP 29: SAVE ORIGINAL HORIZONTAL PROJECTION
    # ========================================================

    projection_v2.save_projection(
        horizontal,
        image_name
        + " Horizontal Histogram Projection",
        "Image Row",
        os.path.join(
            output_folder,
            "horizontal_projection.png"
        )
    )


    # ========================================================
    # STEP 30: SAVE ORIGINAL VERTICAL PROJECTION
    # ========================================================

    projection_v2.save_projection(
        vertical,
        image_name
        + " Vertical Histogram Projection",
        "Image Column",
        os.path.join(
            output_folder,
            "vertical_projection.png"
        )
    )


    # ========================================================
    # STEP 31: SAVE CLEAN HORIZONTAL PROJECTION
    # ========================================================

    projection_v2.save_projection(
        clean_horizontal,
        image_name
        + " Clean Horizontal Projection",
        "Image Row",
        os.path.join(
            output_folder,
            "clean_horizontal_projection.png"
        )
    )


    # ========================================================
    # STEP 32: SAVE CLEAN VERTICAL PROJECTION
    # ========================================================

    projection_v2.save_projection(
        clean_vertical,
        image_name
        + " Clean Vertical Projection",
        "Image Column",
        os.path.join(
            output_folder,
            "clean_vertical_projection.png"
        )
    )


    # ========================================================
    # STEP 33: DISPLAY RESULTS
    # ========================================================

    # Convert original BGR image to RGB.
    image_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Convert detected image to RGB.
    detected_rgb = cv2.cvtColor(
        detected,
        cv2.COLOR_BGR2RGB
    )


    projection_v2.show_image(
        image_rgb,
        "Original Paper - " + image_name
    )

    projection_v2.show_image(
        gray,
        "Grayscale Image - " + image_name,
        "gray"
    )

    projection_v2.show_image(
        binary,
        "Binary Image - " + image_name,
        "gray"
    )

    projection_v2.show_image(
        paragraph_mask,
        "Paragraph Mask - " + image_name,
        "gray"
    )

    projection_v2.show_image(
        detected_rgb,
        "Detected Paragraphs - " + image_name
    )


    # ========================================================
    # STEP 34: DISPLAY HISTOGRAM PROJECTIONS
    # ========================================================

    projection_v2.show_projection(
        horizontal,
        "Horizontal Histogram Projection - "
        + image_name,
        "Image Row"
    )

    projection_v2.show_projection(
        vertical,
        "Vertical Histogram Projection - "
        + image_name,
        "Image Column"
    )


    # ========================================================
    # STEP 35: PRINT RESULT
    # ========================================================

    print("Number of detected paragraphs:", len(sorted_stats))
    print(image_name, "processing complete.")
process_image("001")
process_image("002")
process_image("003")
process_image("004")
process_image("005")
process_image("006")
process_image("007")
process_image("008")
print("\nAll Task B images have been processed.")