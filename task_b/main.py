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


# ============================================================
# STEP 1: READ THE IMAGE
# ============================================================

# Resolve paths relative to this script's own folder, so it works
# whether you run it as `python3 main.py` from inside task_b/, or
# as `python3 task_b/main.py` from the repo root.
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "input", "008.png")
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
print("The otsu threshold value is : ", otsu_value)


# ============================================================
# STEP 4: APPLY BINARY THRESHOLD
# ============================================================

# THRESH_BINARY_INV makes text pixels white (255) and background black (0),
# which is the polarity morphologyEx/dilate expect for "foreground".
threshold_value, binary = cv2.threshold(gray, otsu_value, 255, cv2.THRESH_BINARY_INV)


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

# Create a horizontal structuring element.
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))

# Detect long horizontal lines.
horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)


# ============================================================
# STEP 7: DETECT VERTICAL TABLE LINES
# ============================================================

# Create a vertical structuring element.
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

# Detect long vertical lines.
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

# binary is already text=white/background=black, so this keeps
# only the text pixels that are NOT inside the table region.
text_only = cv2.bitwise_and(binary, inverse_table_mask)


# ============================================================
# STEP 12: CREATE A CLEAN BINARY IMAGE (FOR DISPLAY)
# ============================================================

# Convert back to black-text-on-white for a natural-looking display image.
clean_binary = cv2.bitwise_not(text_only)


# ============================================================
# STEP 13: CLEAN VERTICAL PROJECTION (TABLE REMOVED)
# ============================================================

clean_vertical = projection.vertical_projection(clean_binary)


# ============================================================
# STEP 14: FIND THE GAP BETWEEN COLUMNS (DIAGNOSTIC ONLY)
# ============================================================

# Kept for a quick sanity check while tuning / debugging. The actual
# reading-order sort no longer depends on this single split point --
# see STEP 18, which clusters paragraphs into however many columns
# the page actually has.
column_split = projection.find_column_split(clean_vertical)
print("Column split position (2-column estimate, for reference only):", column_split)


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
print("Number of candidate regions before photo filtering:", len(paragraph_stats))


# ============================================================
# STEP 17b: DROP EMBEDDED PHOTOS (NOT REAL PARAGRAPHS)
# ============================================================

# Some pages have embedded colour photos. Otsu thresholding a photo
# still produces plenty of "dark" pixels (shadows, petal/leaf edges,
# bark texture), so a photo region can accidentally survive the size
# filters above and get treated as a paragraph. Real scientific-paper
# text is essentially black-and-white, while a photo has real colour
# variance, so we use the original colour image's saturation to tell
# them apart: high mean saturation inside a box -> it's a photo, drop it.
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
saturation = hsv[:, :, 1]

SATURATION_THRESHOLD = 15  # text regions measured ~0-1, photo regions ~50-145

is_text = []
for x, y, w, h, area in paragraph_stats:
    mean_saturation = saturation[y : y + h, x : x + w].mean()
    is_text.append(mean_saturation < SATURATION_THRESHOLD)

is_text = np.array(is_text, dtype=bool)
n_dropped = int((~is_text).sum())
if n_dropped:
    print(f"Dropped {n_dropped} region(s) that look like embedded photos, not text.")

paragraph_stats = paragraph_stats[is_text]
print("Number of detected paragraph regions:", len(paragraph_stats))


# ============================================================
# STEP 18: SORT THE PARAGRAPHS INTO READING ORDER
# ============================================================

# Works for any number of columns (single, double, triple, ...) --
# see projection.sort_paragraphs for how the column grouping works.
sorted_stats = projection.sort_paragraphs(paragraph_stats)


# ============================================================
# STEP 19: EXTRACT, DRAW AND SAVE EACH PARAGRAPH
# ============================================================

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

detected = original.copy()

for index, (x, y, w, h, area) in enumerate(sorted_stats, start=1):
    # Crop the paragraph from the original colour image.
    paragraph_crop = original[y : y + h, x : x + w]

    # Save it.
    out_path = os.path.join(output_dir, f"paragraph_{index}.png")
    cv2.imwrite(out_path, paragraph_crop)

    # Draw its bounding box on the detection preview.
    cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 0, 255), 3)

print(f"Saved {len(sorted_stats)} paragraph crops to '{output_dir}'.")


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
# STEP 21: DISPLAY THE ORIGINAL IMAGE
# ============================================================

show_image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original Paper")


# ============================================================
# STEP 22: DISPLAY THE GRAYSCALE IMAGE
# ============================================================

show_image(gray, "Grayscale Image", "gray")


# ============================================================
# STEP 23: DISPLAY THE BINARY IMAGE
# ============================================================

show_image(binary, "Binary Image", "gray")


# ============================================================
# STEP 24: DISPLAY THE PARAGRAPH MASK
# ============================================================

show_image(paragraph_mask, "Paragraph Regions", "gray")


# ============================================================
# STEP 25: DISPLAY THE FINAL RESULT
# ============================================================

show_image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB), "Detected Paragraphs")


# ============================================================
# STEP 26: DISPLAY THE HORIZONTAL PROJECTION
# ============================================================

projection.show_projection(horizontal, "Horizontal Histogram Projection", "Image Row")


# ============================================================
# STEP 27: DISPLAY THE VERTICAL PROJECTION
# ============================================================

projection.show_projection(vertical, "Vertical Histogram Projection", "Image Column")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("Number of extracted paragraphs:", len(sorted_stats))
print("Task B is complete.")

