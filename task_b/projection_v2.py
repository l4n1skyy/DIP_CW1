import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FUNCTION 1: SHOW IMAGE
# ============================================================

def show_image(image, title, colour_map=None):
    """
    Display an image with a title.
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
# FUNCTION 2: SHOW HISTOGRAM PROJECTION
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
# FUNCTION 3: SAVE HISTOGRAM PROJECTION
# ============================================================

def save_projection(projection_values, title, axis_name, filename):
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


# ============================================================
# FUNCTION 4: HORIZONTAL HISTOGRAM PROJECTION
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
# FUNCTION 5: VERTICAL HISTOGRAM PROJECTION
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
# FUNCTION 6: DETECT HORIZONTAL TABLE LINES
# ============================================================

def horizontal_line_detection(binary_image):
    """
    Detect long horizontal lines.

    Morphological opening is performed using:

        erosion
            ↓
        dilation
    """

    # Convert black objects to white foreground.
    foreground = 255 - binary_image

    # Create a horizontal structuring element.
    horizontal_se = np.ones(
        (1, 80),
        dtype=np.uint8
    )

    # First operation of opening: erosion.
    horizontal_eroded = cv2.erode(
        foreground,
        horizontal_se,
        iterations=1
    )

    # Second operation of opening: dilation.
    horizontal_lines = cv2.dilate(
        horizontal_eroded,
        horizontal_se,
        iterations=1
    )

    return horizontal_lines


# ============================================================
# FUNCTION 7: DETECT VERTICAL TABLE LINES
# ============================================================

def vertical_line_detection(binary_image):
    """
    Detect long vertical lines.

    Morphological opening is performed using:

        erosion
            ↓
        dilation
    """

    # Convert black objects to white foreground.
    foreground = 255 - binary_image

    # Create a vertical structuring element.
    vertical_se = np.ones(
        (40, 1),
        dtype=np.uint8
    )

    # First operation of opening: erosion.
    vertical_eroded = cv2.erode(
        foreground,
        vertical_se,
        iterations=1
    )

    # Second operation of opening: dilation.
    vertical_lines = cv2.dilate(
        vertical_eroded,
        vertical_se,
        iterations=1
    )

    return vertical_lines


# ============================================================
# FUNCTION 8: COMBINE TABLE LINES
# ============================================================

def combine_table_lines(horizontal_lines, vertical_lines):
    """
    Combine the detected horizontal and vertical lines.
    """

    table_lines = cv2.add(
        horizontal_lines,
        vertical_lines
    )

    return table_lines


# ============================================================
# FUNCTION 9: REMOVE TABLE LINES
# ============================================================

def remove_table_lines(binary_image, table_lines):
    """
    Remove the detected table lines from the binary image.
    """

    # Convert black text to white foreground.
    foreground = 255 - binary_image

    # Remove the detected table lines.
    clean_foreground = cv2.subtract(
        foreground,
        table_lines
    )

    # Return to black text on a white background.
    clean_binary = 255 - clean_foreground

    return clean_binary


# ============================================================
# FUNCTION 10: CREATE PARAGRAPH MASK
# ============================================================

def create_paragraph_mask(binary_image):
    """
    Use dilation to connect nearby text.

    Nearby letters, words and text lines become
    larger connected regions.
    """

    # Convert black text to white foreground.
    foreground = 255 - binary_image

    # Create a rectangular structuring element.
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