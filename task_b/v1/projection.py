import cv2
import numpy as np
import matplotlib.pyplot as plt


def horizontal_projection(binary_image):
    # count black pixels per row
    return np.count_nonzero(binary_image == 0, axis=1)


def vertical_projection(binary_image):
    # count black pixels per column
    return np.count_nonzero(binary_image == 0, axis=0)


def find_projection_ranges(projection_values):
    active = (projection_values > 0).astype(np.uint8)
    padded = np.pad(active, (1, 1), constant_values=0)
    difference = np.diff(padded)
    start_positions = np.where(difference == 1)[0]
    end_positions = np.where(difference == -1)[0] - 1
    return start_positions, end_positions


def find_column_split(vertical_values):
    # search the middle 40-60% of the page for the gutter (lowest pixel count)
    page_width = len(vertical_values)
    centre_start = int(page_width * 0.40)
    centre_end = int(page_width * 0.60)
    return centre_start + np.argmin(vertical_values[centre_start:centre_end])


def sort_paragraphs(paragraph_stats):
    # sort paragraph x-centres left-to-right, then cut a new column
    # wherever the gap to the next centre is much bigger than a
    # paragraph's own width (a column gutter, not normal spacing)
    x_position = paragraph_stats[:, 0]
    y_position = paragraph_stats[:, 1]
    paragraph_width = paragraph_stats[:, 2]

    x_centre = x_position + (paragraph_width / 2)
    order_by_x = np.argsort(x_centre)
    sorted_centres = x_centre[order_by_x]
    gaps = np.diff(sorted_centres)

    # 80px floor guards narrow columns / single-paragraph pages
    gap_threshold = max(np.median(paragraph_width) * 0.6, 80)

    column_ids_sorted = np.zeros(len(sorted_centres), dtype=int)
    current_column = 0
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            current_column += 1
        column_ids_sorted[i + 1] = current_column

    column_ids = np.zeros(len(x_centre), dtype=int)
    column_ids[order_by_x] = column_ids_sorted

    # column first (left to right), then top-to-bottom within a column
    order = np.lexsort((y_position, column_ids))
    return paragraph_stats[order]


def show_projection(projection_values, title, axis_name):
    plt.figure(figsize=(10, 4))
    plt.plot(projection_values)
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel("Number of Black Text Pixels")
    plt.tight_layout()
    plt.show()
