import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

import projection  # this imports the projection.py file with functions in the same folder


## function to show image (so code is more efficient a bit)
def show_image(image, title, color_map=None):
    plt.figure(figsize=(10, 12))
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def process_image(image_path, output_dir, show_plots=False):
    """
    Runs the full paragraph-extraction pipeline on a single image and
    saves all outputs (paragraph crops + intermediate processing images)
    into output_dir.
    """

    print(f"Processing: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # STEP 1: READ THE IMAGE
    # ============================================================

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Keep an untouched colour copy to crop the final paragraphs from.
    original = image.copy()

    # ============================================================
    # STEP 2: CONVERT TO GRAYSCALE
    # ============================================================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ============================================================
    # STEP 3: FIND OTSU THRESHOLD
    # ============================================================

    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ============================================================
    # STEP 4: APPLY BINARY THRESHOLD
    # ============================================================

    # THRESH_BINARY_INV makes text pixels white (255) and background black (0),
    # which is the polarity morphologyEx/dilate expect for "foreground".
    threshold_value, binary = cv2.threshold(
        gray, otsu_value, 255, cv2.THRESH_BINARY_INV
    )

    # ============================================================
    # STEP 5: HISTOGRAM PROJECTIONS
    # ============================================================

    # counts the number of black text pixels in every image row and column
    horizontal = projection.horizontal_projection(binary)
    vertical = projection.vertical_projection(binary)

    # finding the starting and ending row text ranges
    row_start, row_end = projection.find_projection_ranges(horizontal)
    # finding the starting and ending column text ranges
    column_start, column_end = projection.find_projection_ranges(vertical)

    # ============================================================
    # STEP 6: DETECT HORIZONTAL TABLE LINES
    # ============================================================

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # ============================================================
    # STEP 7: DETECT VERTICAL TABLE LINES
    # ============================================================

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # ============================================================
    # STEP 8: COMBINE THE TABLE LINES
    # ============================================================

    table_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # ============================================================
    # STEP 9: FIND THE TABLE REGION
    # ============================================================

    table_contours, hierarchy = cv2.findContours(
        table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ============================================================
    # STEP 10: CREATE THE TABLE MASK
    # ============================================================

    table_mask = np.zeros_like(binary)
    cv2.drawContours(table_mask, table_contours, -1, 255, cv2.FILLED)

    # ============================================================
    # STEP 11: REMOVE THE TABLE FROM THE TEXT MASK
    # ============================================================

    inverse_table_mask = cv2.bitwise_not(table_mask)
    text_only = cv2.bitwise_and(binary, inverse_table_mask)

    # ============================================================
    # STEP 12: CREATE A CLEAN BINARY IMAGE (FOR DISPLAY)
    # ============================================================

    clean_binary = cv2.bitwise_not(text_only)

    # ============================================================
    # STEP 13: CLEAN VERTICAL PROJECTION (TABLE REMOVED)
    # ============================================================

    clean_vertical = projection.vertical_projection(clean_binary)

    # ============================================================
    # STEP 14: FIND THE GAP BETWEEN COLUMNS (DIAGNOSTIC ONLY)
    # ============================================================

    column_split = projection.find_column_split(clean_vertical)

    # ============================================================
    # STEP 15: CONNECT TEXT INTO PARAGRAPH REGIONS
    # ============================================================

    paragraph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 20))
    paragraph_mask = cv2.dilate(text_only, paragraph_kernel, iterations=1)

    # ============================================================
    # STEP 16: FIND CONNECTED PARAGRAPH REGIONS
    # ============================================================

    number_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        paragraph_mask, connectivity=8
    )

    # Remove the background component (label 0).
    stats = stats[1:]

    # ============================================================
    # STEP 17: REMOVE SMALL / NOISE REGIONS
    # ============================================================

    valid = (
        (stats[:, cv2.CC_STAT_WIDTH] > 250)
        & (stats[:, cv2.CC_STAT_HEIGHT] > 50)
        & (stats[:, cv2.CC_STAT_AREA] > 10000)
    )

    paragraph_stats = stats[valid]

    # ============================================================
    # STEP 17b: DROP EMBEDDED PHOTOS (NOT REAL PARAGRAPHS)
    # ============================================================

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    SATURATION_THRESHOLD = 15  # text regions measured ~0-1, photo regions ~50-145

    is_text = []
    for x, y, w, h, area in paragraph_stats:
        mean_saturation = saturation[y : y + h, x : x + w].mean()
        is_text.append(mean_saturation < SATURATION_THRESHOLD)

    is_text = np.array(is_text, dtype=bool)
    paragraph_stats = paragraph_stats[is_text]

    # ============================================================
    # STEP 18: SORT THE PARAGRAPHS INTO READING ORDER
    # ============================================================

    if len(paragraph_stats) == 0:
        sorted_stats = paragraph_stats
    else:
        sorted_stats = projection.sort_paragraphs(paragraph_stats)

    # ============================================================
    # STEP 19: EXTRACT, DRAW AND SAVE EACH PARAGRAPH
    # ============================================================

    detected = original.copy()

    for index, (x, y, w, h, area) in enumerate(sorted_stats, start=1):
        # Crop the paragraph from the original colour image.
        paragraph_crop = original[y : y + h, x : x + w]

        # Save it.
        out_path = os.path.join(output_dir, f"paragraph_{index}.png")
        cv2.imwrite(out_path, paragraph_crop)

        # Draw its bounding box on the detection preview.
        cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # ============================================================
    # STEP 20: SAVE THE PROCESSING OUTPUTS
    # ============================================================

    cv2.imwrite(os.path.join(output_dir, "binary.png"), binary)
    cv2.imwrite(os.path.join(output_dir, "table_lines.png"), table_lines)
    cv2.imwrite(os.path.join(output_dir, "table_mask.png"), table_mask)
    cv2.imwrite(os.path.join(output_dir, "clean_binary.png"), clean_binary)
    cv2.imwrite(os.path.join(output_dir, "paragraph_mask.png"), paragraph_mask)
    cv2.imwrite(os.path.join(output_dir, "detected_paragraphs.png"), detected)

    # ============================================================
    # STEPS 21-27: DISPLAY (optional, off by default when batch processing)
    # ============================================================

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


# ============================================================
# MAIN: RUN OVER EVERY IMAGE IN input/
# ============================================================

if __name__ == "__main__":
    # Resolve paths relative to this script's own folder, so it works
    # whether you run it as `python3 main.py` from inside task_b/, or
    # as `python3 task_b/main.py` from the repo root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_root = os.path.join(script_dir, "output")

    # Set to True if you want the matplotlib windows to pop up for every
    # image (slow / annoying for a batch run) -- False just saves files.
    SHOW_PLOTS = False

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
        # e.g. input/001.png -> output/001/
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_root, name)

        count = process_image(image_path, output_dir, show_plots=SHOW_PLOTS)
        summary[name] = count

    print("\nALL IMAGES PROCESSED")
    for name, count in summary.items():
        print(f"  {name}: {count} paragraph(s) extracted -> output/{name}/")
