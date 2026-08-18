import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

import projection


def show_image(image, title, color_map=None):
    plt.figure(figsize=(10, 12))
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def process_image(image_path, output_dir, show_plots=False):
    """Runs the paragraph-extraction pipeline on one image and saves
    the crops + intermediate images into output_dir."""

    print(f"Processing: {image_path}")
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # keep an untouched colour copy to crop paragraphs from
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # INV makes text=white(255), background=black(0) -- the polarity
    # morphologyEx/dilate expect for "foreground"
    _, binary = cv2.threshold(gray, otsu_value, 255, cv2.THRESH_BINARY_INV)

    horizontal = projection.horizontal_projection(binary)
    vertical = projection.vertical_projection(binary)
    row_start, row_end = projection.find_projection_ranges(horizontal)
    column_start, column_end = projection.find_projection_ranges(vertical)

    # detect table grid lines so they can be masked out before paragraph detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    table_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # fill the table's bounding region solid so it can be removed entirely
    table_contours, hierarchy = cv2.findContours(
        table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    table_mask = np.zeros_like(binary)
    cv2.drawContours(table_mask, table_contours, -1, 255, cv2.FILLED)

    inverse_table_mask = cv2.bitwise_not(table_mask)
    text_only = cv2.bitwise_and(binary, inverse_table_mask)
    clean_binary = cv2.bitwise_not(text_only)  # black-on-white, for display

    clean_vertical = projection.vertical_projection(clean_binary)
    column_split = projection.find_column_split(clean_vertical)  # diagnostic only

    # dilate to connect nearby text into paragraph blobs
    paragraph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 30))
    paragraph_mask = cv2.dilate(text_only, paragraph_kernel, iterations=1)

    number_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        paragraph_mask, connectivity=8
    )
    stats = stats[1:]  # drop background component (label 0)

    valid = (
        (stats[:, cv2.CC_STAT_WIDTH] > 250)
        & (stats[:, cv2.CC_STAT_HEIGHT] > 50)
        & (stats[:, cv2.CC_STAT_AREA] > 10000)
    )
    paragraph_stats = stats[valid]

    # drop embedded photos: text has near-zero saturation, photos don't
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    SATURATION_THRESHOLD = 15

    is_text = []
    for x, y, w, h, area in paragraph_stats:
        mean_saturation = saturation[y : y + h, x : x + w].mean()
        is_text.append(mean_saturation < SATURATION_THRESHOLD)
    is_text = np.array(is_text, dtype=bool)
    paragraph_stats = paragraph_stats[is_text]

    if len(paragraph_stats) == 0:
        sorted_stats = paragraph_stats
    else:
        sorted_stats = projection.sort_paragraphs(paragraph_stats)

    detected = original.copy()
    for index, (x, y, w, h, area) in enumerate(sorted_stats, start=1):
        paragraph_crop = original[y : y + h, x : x + w]
        out_path = os.path.join(output_dir, f"paragraph_{index}.png")
        cv2.imwrite(out_path, paragraph_crop)
        cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imwrite(os.path.join(output_dir, "binary.png"), binary)
    cv2.imwrite(os.path.join(output_dir, "table_lines.png"), table_lines)
    cv2.imwrite(os.path.join(output_dir, "table_mask.png"), table_mask)
    cv2.imwrite(os.path.join(output_dir, "clean_binary.png"), clean_binary)
    cv2.imwrite(os.path.join(output_dir, "paragraph_mask.png"), paragraph_mask)
    cv2.imwrite(os.path.join(output_dir, "detected_paragraphs.png"), detected)

    if show_plots:
        show_image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original Paper")
        show_image(gray, "Grayscale Image", "gray")
        show_image(binary, "Binary Image", "gray")
        show_image(paragraph_mask, "Paragraph Regions", "gray")
        show_image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB), "Detected Paragraphs")
        projection.show_projection(
            horizontal, "Horizontal Histogram Projection", "Image Row"
        )
        projection.show_projection(
            vertical, "Vertical Histogram Projection", "Image Column"
        )

    return len(sorted_stats)


if __name__ == "__main__":
    # resolve paths relative to this file so it works from any cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_root = os.path.join(script_dir, "output")

    SHOW_PLOTS = False  # set True to pop up matplotlib windows per image

    all_files = sorted(os.listdir(input_dir))
    valid_extensions = (".png", ".jpg", ".jpeg")
    image_paths = [
        os.path.join(input_dir, filename)
        for filename in all_files
        if filename.lower().endswith(valid_extensions)
    ]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    summary = {}
    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_root, name)
        count = process_image(image_path, output_dir, show_plots=SHOW_PLOTS)
        summary[name] = count

    print("\nALL IMAGES PROCESSED")
    for name, count in summary.items():
        print(f"  {name}: {count} paragraph(s) extracted -> output/{name}/")
