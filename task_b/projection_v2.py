# -*- coding: utf-8 -*-

"""
Task B - Paragraph Extraction Functions

This file contains reusable functions for:

1. Horizontal histogram projection
2. Vertical histogram projection
3. Horizontal table line detection
4. Vertical table line detection
5. Table line removal
6. Paragraph region creation
7. Paragraph sorting
8. Image and projection display

Binary image format:

    Text       = Black = 0
    Background = White = 255
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FUNCTION 1: HORIZONTAL HISTOGRAM PROJECTION
# ============================================================

def horizontal_projection(binary_image):
    """
    Count the number of black pixels in every image row.
    """

    horizontal = np.count_nonzero(
        binary_image == 0,
        axis=1
    )

    return horizontal


# ============================================================
# FUNCTION 2: VERTICAL HISTOGRAM PROJECTION
# ============================================================

def vertical_projection(binary_image):
    """
    Count the number of black pixels in every image column.
    """

    vertical = np.count_nonzero(
        binary_image == 0,
        axis=0
    )

    return vertical


# ============================================================
# FUNCTION 3: DETECT HORIZONTAL TABLE LINES
# ============================================================

def horizontal_line_detection(binary_image):
    """
    Detect long horizontal table lines.

    Morphological opening is performed using:

        Erosion
           ↓
        Dilation
    """

    # Convert black objects into white foreground.
    foreground = 255 - binary_image

    # Create horizontal structuring element.
    horizontal_se = np.ones(
        (1, 80),
        dtype=np.uint8
    )

    # Erosion.
    horizontal_eroded = cv2.erode(
        foreground,
        horizontal_se,
        iterations=1
    )

    # Dilation.
    horizontal_lines = cv2.dilate(
        horizontal_eroded,
        horizontal_se,
        iterations=1
    )

    return horizontal_lines


# ============================================================
# FUNCTION 4: DETECT VERTICAL TABLE LINES
# ============================================================

def vertical_line_detection(binary_image):
    """
    Detect long vertical table lines.

    Morphological opening is performed using:

        Erosion
           ↓
        Dilation
    """

    # Convert black objects into white foreground.
    foreground = 255 - binary_image

    # Create vertical structuring element.
    vertical_se = np.ones(
        (40, 1),
        dtype=np.uint8
    )

    # Erosion.
    vertical_eroded = cv2.erode(
        foreground,
        vertical_se,
        iterations=1
    )

    # Dilation.
    vertical_lines = cv2.dilate(
        vertical_eroded,
        vertical_se,
        iterations=1
    )

    return vertical_lines


# ============================================================
# FUNCTION 5: COMBINE TABLE LINES
# ============================================================

def combine_table_lines(horizontal_lines, vertical_lines):
    """
    Combine detected horizontal and vertical table lines.
    """

    table_lines = cv2.add(
        horizontal_lines,
        vertical_lines
    )

    return table_lines


# ============================================================
# FUNCTION 6: REMOVE TABLE LINES
# ============================================================

def remove_table_lines(binary_image, table_lines):
    """
    Remove detected table lines from the binary image.
    """

    # Convert black text into white foreground.
    foreground = 255 - binary_image

    # Remove the detected table lines.
    clean_foreground = cv2.subtract(
        foreground,
        table_lines
    )

    # Return to black text on white background.
    clean_binary = 255 - clean_foreground

    return clean_binary


# ============================================================
# FUNCTION 7: CREATE PARAGRAPH MASK
# ============================================================

def create_paragraph_mask(binary_image):
    """
    Use dilation to connect nearby text.

    Nearby letters, words and text lines become
    larger connected paragraph regions.
    """

    # Convert black text into white foreground.
    foreground = 255 - binary_image

    # Create rectangular structuring element.
    paragraph_se = np.ones(
        (20, 25),
        dtype=np.uint8
    )

    # Connect nearby text.
    paragraph_mask = cv2.dilate(
        foreground,
        paragraph_se,
        iterations=1
    )

    return paragraph_mask


# ============================================================
# FUNCTION 8: SORT PARAGRAPHS
# ============================================================

def sort_paragraphs(paragraph_stats):
    """
    Sort paragraph regions into reading order.

    The paragraph positions are first analysed
    from left to right.

    Large horizontal gaps are treated as gaps
    between document columns.

    Paragraphs are then sorted:
        1. Left column to right column
        2. Top to bottom inside each column
    """

    # Get x positions.
    x_position = paragraph_stats[:, 0]

    # Get y positions.
    y_position = paragraph_stats[:, 1]

    # Get paragraph widths.
    paragraph_width = paragraph_stats[:, 2]


    # --------------------------------------------------------
    # Calculate horizontal centre of every paragraph.
    # --------------------------------------------------------

    x_centre = (
        x_position
        + paragraph_width / 2
    )


    # --------------------------------------------------------
    # Sort paragraph centres from left to right.
    # --------------------------------------------------------

    x_order = np.argsort(
        x_centre
    )

    sorted_centres = x_centre[
        x_order
    ]


    # --------------------------------------------------------
    # Calculate gaps between paragraph centres.
    # --------------------------------------------------------

    centre_gaps = np.diff(
        sorted_centres
    )


    # --------------------------------------------------------
    # Estimate the gap between document columns.
    # --------------------------------------------------------

    gap_threshold = max(
        np.median(paragraph_width) * 0.6,
        80
    )


    # --------------------------------------------------------
    # Create column numbers.
    # --------------------------------------------------------

    sorted_columns = np.zeros(
        len(sorted_centres),
        dtype=int
    )

    column_number = 0


    # --------------------------------------------------------
    # Detect large gaps between columns.
    # --------------------------------------------------------

    for position, gap in enumerate(centre_gaps):

        if gap > gap_threshold:

            column_number = (
                column_number + 1
            )

        sorted_columns[
            position + 1
        ] = column_number


    # --------------------------------------------------------
    # Put column numbers back into original order.
    # --------------------------------------------------------

    column_ids = np.zeros(
        len(x_centre),
        dtype=int
    )

    column_ids[
        x_order
    ] = sorted_columns


    # --------------------------------------------------------
    # Sort by column first and y position second.
    # --------------------------------------------------------

    reading_order = np.lexsort(
        (
            y_position,
            column_ids
        )
    )

    sorted_stats = paragraph_stats[
        reading_order
    ]

    return sorted_stats


# ============================================================
# FUNCTION 9: SHOW IMAGE
# ============================================================

def show_image(image, title, colour_map=None):
    """
    Display an image using Matplotlib.
    """

    plt.figure(
        figsize=(10, 12)
    )

    plt.imshow(
        image,
        cmap=colour_map
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# FUNCTION 10: SHOW HISTOGRAM PROJECTION
# ============================================================

def show_projection(projection_values, title, axis_name):
    """
    Display a histogram projection.
    """

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        projection_values
    )

    plt.title(
        title
    )

    plt.xlabel(
        axis_name
    )

    plt.ylabel(
        "Number of Black Pixels"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# FUNCTION 11: SAVE HISTOGRAM PROJECTION
# ============================================================

def save_projection(
        projection_values,
        title,
        axis_name,
        filename
):
    """
    Save a histogram projection as an image.
    """

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        projection_values
    )

    plt.title(
        title
    )

    plt.xlabel(
        axis_name
    )

    plt.ylabel(
        "Number of Black Pixels"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()