"""This was coded in vs code using anaconda interpreter so that the code can be used for colaboration - shi feng & lani """

"""Task B - Paragraph Extraction Functions"""

#imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

def horizontal_projection(binary_image):
    horizontal = np.count_nonzero(binary_image == 0,axis=1) #counts the number of black pixels in every row
    return horizontal

def vertical_projection(binary_image):
    vertical = np.count_nonzero(binary_image == 0,axis=0) #counts the number of black pixels in every column
    return vertical

#this section uses dilation and erosion to determine horizontal lines
def horizontal_line_detection(binary_image):

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


#these codes are to detect vertical table lines using erosion then dilation
def vertical_line_detection(binary_image):
    # Convert black objects into white foreground.
    foreground = 255 - binary_image

    # Create vertical structuring element.
    vertical_se = np.ones(
        (40, 1),
        dtype=np.uint8
    )

    #Erosion segment
    vertical_eroded = cv2.erode(
        foreground,
        vertical_se,
        iterations=1
    )

    #Dilation segment
    vertical_lines = cv2.dilate(
        vertical_eroded,
        vertical_se,
        iterations=1
    )
    return vertical_lines

#this snippet of code combines the 
#detected horizontal and vertical lines
def combine_table_lines(horizontal_lines, vertical_lines):
    """
    Combine detected horizontal and vertical table lines.
    """

    table_lines = cv2.add(
        horizontal_lines,
        vertical_lines
    )

    return table_lines

#this creates the full table region
def create_table_region_mask(table_lines):
    stats= (cv2.connectedComponentsWithStats(table_lines,connectivity=8))  #finds the connected table-line regions.
    stats = stats[1:]                                                      #this removes the background.
    page_height, page_width = table_lines.shape                            #gets the page size

    #this snippet of code ensures that the table stats are determined 
    valid_tables = (
        (stats[:, cv2.CC_STAT_WIDTH]
         > page_width * 0.15)
        &
        (stats[:, cv2.CC_STAT_HEIGHT]
         > page_height * 0.02)
    )

    table_stats = stats[valid_tables]

    #creates empty table mask
    table_mask = np.zeros_like(
        table_lines
    )

    #fills the detected table area with white colour
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

def remove_table_region(binary_image,table_mask):
    # Convert text into white foreground.
    foreground = 255 - binary_image

    #removes everything inside of table area
    clean_foreground = cv2.subtract(foreground,table_mask)

    #returns to black text on white background.
    clean_binary = 255 - clean_foreground

    return clean_binary


#clean parahraph mask section (uses dilation again)
def create_paragraph_mask(binary_image):
    #converts black text into white foreground
    foreground = 255 - binary_image

    #creates rectangular structuring element (red box thing)
    paragraph_se = np.ones(
        (30, 25),
        dtype=np.uint8
        )

    #connects all nearby text
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
        saturation_threshold=15):
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

        #lower saturated areas are likely to be text within detected region
        is_text.append(mean_saturation< saturation_threshold)

    #this converts the results into a NumPy Boolean array, for what i have no clue
    is_text = np.array(is_text, dtype=bool)

    #it keeps only the regions that are classified as text
    filtered_stats = paragraph_stats[is_text]

    return filtered_stats

#sorting segment , some sorcery
def sort_paragraphs(paragraph_stats):

    #gets x and y positions of columns , and the width of each paragraph
    x_position = paragraph_stats[:, 0]
    y_position = paragraph_stats[:, 1]
    paragraph_width = paragraph_stats[:, 2]

    #this determines the center of each paragraph detected
    x_centre = (x_position+ paragraph_width / 2)

    #this segment sorts the paragrapghs from left side to right side so that it is consistent
    x_order = np.argsort(x_centre)
    sorted_centres = x_centre[x_order]

    #this determines the amount of gap between each paragrapgh sections
    centre_gaps = np.diff(sorted_centres)
    gap_threshold = max(np.median(paragraph_width) * 0.6,80)

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

def show_projection(projection_values, title, axis_name):
    plt.figure(figsize=(10, 4))
    plt.plot(projection_values)
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel("Number of Black Pixels")
    plt.tight_layout()
    plt.show()