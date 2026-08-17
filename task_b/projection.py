# This file contains the functions
# used for histogram projections,
# so the main.py file is more efficient.
import numpy as np
import matplotlib.pyplot as plt


# horizontal projection
def horizontal_projection(binary_image):
    horizontal = np.count_nonzero(binary_image == 0, axis=1)
    return horizontal


# vertical projection
def vertical_projection(binary_image):
    vertical = np.count_nonzero(binary_image == 0, axis=0)
    return vertical


# find text ranges
def find_projection_ranges(projection_values):
    active = (projection_values > 0).astype(
        np.uint8
    )  # converts projection into active & inactive positions
    padded = np.pad(
        active, (1, 1), constant_values=0
    )  # padding for before and after projection
    difference = np.diff(padded)  # finds changes between active and inactive positions
    start_positions = np.where(difference == 1)[
        0
    ]  # change from 0 --> 1 represents starting position
    end_positions = (
        np.where(difference == -1)[0] - 1
    )  # change from 1 --> 0 represents ending position
    return start_positions, end_positions


# column split shenanigans
def find_column_split(vertical_values):
    page_width = len(vertical_values)  # getting the width of the projection
    # search the center of the page
    centre_start = int(page_width * 0.40)
    centre_end = int(page_width * 0.60)
    # finding the position with the lowest amount of pixels (the gutter)
    column_split = centre_start + np.argmin(vertical_values[centre_start:centre_end])
    return column_split


# sorting functions
def sort_paragraphs(paragraph_stats):
    # Works for any number of columns (single, double, triple, ...).
    # Idea: sort paragraph x-centres left-to-right, then cut a new
    # "column" wherever the gap to the next x-centre is much bigger
    # than a paragraph's own width -- that gap is a column gutter,
    # not just normal paragraph-to-paragraph variation.
    x_position = paragraph_stats[:, 0]
    y_position = paragraph_stats[:, 1]
    paragraph_width = paragraph_stats[:, 2]

    # calculate the centre of each paragraph
    x_centre = x_position + (paragraph_width / 2)

    order_by_x = np.argsort(x_centre)
    sorted_centres = x_centre[order_by_x]

    gaps = np.diff(sorted_centres)
    # a gutter gap should be clearly bigger than a paragraph's own width;
    # 80px floor guards against very narrow columns / single-paragraph pages
    gap_threshold = max(np.median(paragraph_width) * 0.6, 80)

    column_ids_sorted = np.zeros(len(sorted_centres), dtype=int)
    current_column = 0
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            current_column += 1
        column_ids_sorted[i + 1] = current_column

    # map column ids back to the original (unsorted) row order
    column_ids = np.zeros(len(x_centre), dtype=int)
    column_ids[order_by_x] = column_ids_sorted

    # column first (left to right), then top-to-bottom within a column
    order = np.lexsort((y_position, column_ids))
    sorted_stats = paragraph_stats[order]
    return sorted_stats


# projection display
def show_projection(projection_values, title, axis_name):
    plt.figure(figsize=(10, 4))
    plt.plot(projection_values)
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel("Number of Black Text Pixels")
    plt.tight_layout()
    plt.show()

