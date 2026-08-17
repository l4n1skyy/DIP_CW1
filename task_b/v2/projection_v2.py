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
# CREATE FULL TABLE REGION
# ============================================================

def create_table_region_mask(table_lines):
    """
    Detect the complete area occupied by a table.

    The full table area is filled so that both
    the table lines and the text inside the table
    can be removed.
    """

    # Find connected table-line regions.
    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            table_lines,
            connectivity=8
        )
    )

    # Remove background.
    stats = stats[1:]

    # Get page size.
    page_height, page_width = table_lines.shape

    # A table should have noticeable width and height.
    valid_tables = (
        (stats[:, cv2.CC_STAT_WIDTH]
         > page_width * 0.15)
        &
        (stats[:, cv2.CC_STAT_HEIGHT]
         > page_height * 0.02)
    )

    table_stats = stats[
        valid_tables
    ]

    # Create empty table mask.
    table_mask = np.zeros_like(
        table_lines
    )

    # Fill each detected table area.
    for table in table_stats:

        x = table[
            cv2.CC_STAT_LEFT
        ]

        y = table[
            cv2.CC_STAT_TOP
        ]

        width = table[
            cv2.CC_STAT_WIDTH
        ]

        height = table[
            cv2.CC_STAT_HEIGHT
        ]

        cv2.rectangle(
            table_mask,
            (x, y),
            (
                x + width - 1,
                y + height - 1
            ),
            255,
            -1
        )

    return table_mask


# ============================================================
# REMOVE FULL TABLE REGION
# ============================================================

def remove_table_region(
        binary_image,
        table_mask
):
    """
    Remove both the table lines and the
    text inside the table.
    """

    # Convert text into white foreground.
    foreground = 255 - binary_image

    # Remove everything inside table area.
    clean_foreground = cv2.subtract(
        foreground,
        table_mask
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
        (30, 25),
        dtype=np.uint8
    )

    # Connect nearby text.
    paragraph_mask = cv2.dilate(
        foreground,
        paragraph_se,
        iterations=1
    )

    return paragraph_mask

#bug fix function for images
def remove_color_regions(
        original_image,
        paragraph_stats,
        saturation_threshold=15
):
    #this converts the images in #png 004 and 007 to HSV format
    hsv_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)

    saturation = hsv_image[:,:,1]#what.?

    #store data into array or something
    is_text = [] #feels like javascript

    #check detected paragragh regions
    for paragraph in paragraph_stats:

        #these get the paragraph position
        x=paragraph[cv2.CC_STAT_LEFT]
        y=paragraph[cv2.CC_STAT_TOP]

        #these gets the paragraph size
        width = paragraph[cv2.CC_STAT_WIDTH]
        height = paragraph[cv2.CC_STAT_HEIGHT]

        #these gets the saturation value inside detected region
        region_saturation = saturation[y:y + height, x:x + width]

        #this calculates the average saturation.
        mean_saturation = np.mean(
            region_saturation
        )

        # Low saturation is more likely to be text.
        is_text.append(
            mean_saturation
            < saturation_threshold
        )

    # Convert the results into a NumPy Boolean array.
    is_text = np.array(is_text, dtype=bool)

    # Keep only the regions classified as text.
    filtered_stats = paragraph_stats[is_text]

    return filtered_stats

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

    sorted_columns = np.zeros(len(sorted_centres),dtype=int)
    column_number = 0

    #the following codes are used to detect large gaps between columns
    for position, gap in enumerate(centre_gaps):

        if gap > gap_threshold:
            
            column_number = (column_number + 1)
            
        sorted_columns[position + 1] = column_number

    #puts numbers into columns
    column_ids = np.zeros(len(x_centre),dtype=int)
    column_ids[x_order] = sorted_columns

    #these make sure that the code sorts by columns first
    reading_order = np.lexsort((y_position,column_ids))
    sorted_stats = paragraph_stats[ reading_order]
    return sorted_stats


def show_image(image, title, colour_map=None):

    plt.figure(figsize=(10, 12))

    plt.imshow(image,cmap=colour_map)

    plt.title(title)

    plt.axis("off")

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