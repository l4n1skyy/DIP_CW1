# -*- coding: utf-8 -*-
import os
import cv2
import projection_v2

#Get the folder where main.py is located.
task_b_folder = os.path.dirname(
    os.path.abspath(__file__)
)


#file path opening shenanigans using os
def process_image(image_name):
    print("\nProcessing:",image_name + ".png")

    image_path = os.path.join(task_b_folder,"input",image_name + ".png")

    #reding the iamge
    image = cv2.imread(image_path)
    #keeping the original image for paragraph cropping.
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

    table_mask = projection_v2.create_table_region_mask(
        table_lines
    )

    # ========================================================
    # STEP 12: REMOVE TABLE LINES
    # ========================================================

    clean_binary = projection_v2.remove_table_region(binary,table_mask)


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
    paragraph_stats = projection_v2.remove_color_regions(
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
        # Convert extracted paragraph to RGB.
        paragraph_rgb = cv2.cvtColor(
            paragraph_image,
            cv2.COLOR_BGR2RGB
        )

        # Display extracted paragraph.
        projection_v2.show_image(
            paragraph_rgb,
            "Paragraph "
            + str(paragraph_number)
            + " - "
            + image_name
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
    #==============
    #output segment
    #==============

    #ommmited declarations
    # Convert original image from BGR to RGB for Matplotlib.
    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Convert detected image from BGR to RGB for Matplotlib.
    detected_rgb = cv2.cvtColor(
        detected,
        cv2.COLOR_BGR2RGB
    )
        # Original paper.
    projection_v2.show_image(
            original_rgb,
            "Original Paper - " + image_name
        )


    # Grayscale image.
    projection_v2.show_image(
        gray,
        "Grayscale Image - " + image_name,
        "gray"
    )


    # Binary image.
    projection_v2.show_image(
        binary,
        "Binary Image - " + image_name,
        "gray"
    )


    # Horizontal and vertical table lines combined.
    projection_v2.show_image(
        table_lines,
        "Detected Table Lines - " + image_name,
        "gray"
    )


    # Complete table regions.
    projection_v2.show_image(
        table_mask,
        "Detected Table Region - " + image_name,
        "gray"
    )


    # Image after tables are removed.
    projection_v2.show_image(
        clean_binary,
        "Image After Table Removal - " + image_name,
        "gray"
    )


    # Paragraph mask after dilation.
    projection_v2.show_image(
        paragraph_mask,
        "Paragraph Mask - " + image_name,
        "gray"
    )


    # Final detected paragraphs.
    projection_v2.show_image(
        detected_rgb,
        "Detected Paragraphs - " + image_name
    )


    # ========================================================
    # STEP 24: DISPLAY HISTOGRAM PROJECTIONS
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