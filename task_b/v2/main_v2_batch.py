# -*- coding: utf-8 -*-

"""this is the main code to run to precess the images in batch ,a single use version was scraped"""
import cv2
import numpy as np
import projection_v2


#everything in as a main process
def process_image(image_name):
    image_path = "task_b/v2/input/" + image_name + ".png"     #gets image from folder
    image = cv2.imread(image_path)                            #read image

    if image is None:
        print("Error: Could not read", image_path)  #error catching for empty / not valid images
        return
    
    original = image.copy()                         #keeping the original image for paragraph cropping.

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   #makes image grayscale

    #Finding otsu threshold
    #Otsu automatically finds the threshold value
    otsu_value , otsu_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    print("Otsu threshold:", otsu_value)

    #the text becomes black = 0.
    #background becomes white = 255.
    _, binary = cv2.threshold(gray,otsu_value,255,cv2.THRESH_BINARY) #binary threshold shenanigans

    #histogram projection
    horizontal = projection_v2.horizontal_projection(binary)
    vertical = projection_v2.vertical_projection(binary)
    horizontal_lines = (projection_v2.horizontal_line_detection(binary))
    vertical_lines = (projection_v2.vertical_line_detection(binary))


    # table detection segment + removing table lines 
    # because there was a bug for the red rectangle generation where small random gaps 
    # were detected as a paragrapgh making the code not accurate
    table_lines = projection_v2.combine_table_lines(horizontal_lines,vertical_lines)
    table_mask = projection_v2.create_table_region_mask(table_lines)
    clean_binary = projection_v2.remove_table_region(binary,table_mask)

    #paragrapgh detection segment
    paragraph_mask = projection_v2.create_paragraph_mask(clean_binary)
    _, _, stats, _ = (cv2.connectedComponentsWithStats(paragraph_mask,connectivity=8))
    stats = stats[1:]


    #noise removal section
    #used because the code did bug out and detected every single group of text as a paragrapgh without this (GeeksForGeeks, 2026)
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

#bug fix for color images (GeeksForGeeks, 2026)
    paragraph_stats = projection_v2.remove_color_regions(
        original,
        paragraph_stats,
        saturation_threshold=15)

    #sorting the paragrapghs (ANOTHER SEQUENCE FIX AAA)
    sorted_stats = projection_v2.sort_paragraphs(paragraph_stats) #this sorts the paragrapgh from projection_v2.py
    detected = original.copy() #this prepares the detected image for display
    #storage for sorted paragrapghs , a sequnce error was made when the output was out  
    # becasue original images first and then paragrapgh instead of paragrapgh fisrt
    paragraph_images = [] #array or list to temporarily stoer paragrapghs

    #paragrapgh extraction , a loop is used to reduce redundant code
    for paragraph_number, paragraph in enumerate(
    sorted_stats,
    start=1):

        #gets the position of each paragrapgh
        x = paragraph[cv2.CC_STAT_LEFT]
        y = paragraph[ cv2.CC_STAT_TOP]
        #determines paragrapgh size
        width = paragraph[cv2.CC_STAT_WIDTH]
        height = paragraph[cv2.CC_STAT_HEIGHT]

        #crops the paragrapgh size (GeeksForGeeks, 2026)
        paragraph_image = original[
            y:y + height,
            x:x + width
        ]

        #storage for paragrapgh for display later (sequence bug fix)
        paragraph_images.append((paragraph_number, paragraph_image))

        #this code is used to make the red outer bound rectangle
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
    
    #Convert original image from BGR to RGB for Matplotlib
    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    #Convert detected image from BGR to RGB for Matplotlib
    detected_rgb = cv2.cvtColor(
        detected,
        cv2.COLOR_BGR2RGB
    )

    #==============
    #output segment
    #==============

    #Shows the original paper
    projection_v2.show_image(
            original_rgb,
            "Original Paper - " + image_name
        )

    #Shows the grayscale image
    projection_v2.show_image(
        gray,
        "Grayscale Image - " + image_name,
        "gray"
    )

    #Shows the otsu threshold image
    projection_v2.show_image(otsu_image, "Otsu Threshold Image" + image_name)

    #Shows the binary image
    projection_v2.show_image(
        binary,
        "Binary Image - " + image_name,
        "gray"
    )

    #Displaying the histogram projections
    #horizontal histogram
    projection_v2.show_projection(
        horizontal,
        "Horizontal Histogram Projection - "
        + image_name,
        "Image Row"
    )

    #vertical histogram
    projection_v2.show_projection(
        vertical,
        "Vertical Histogram Projection - "
        + image_name,
        "Image Column"
    )

    #Shows the detected horizontal and vertical table lines combined (if any , for all images)
    projection_v2.show_image(
        table_lines,
        "Detected Table Lines - " + image_name,
        "gray"
    )


    #shows the table line regions (shows the whole area to be deleted / excluded)
    projection_v2.show_image(
        table_mask,
        "Detected Table Region - " + image_name,
        "gray"
    )


    #shows the image after tables are removed
    projection_v2.show_image(
        clean_binary,
        "Image After Table Removal - " + image_name,
        "gray"
    )


    #Paragraph mask after dilation
    projection_v2.show_image(
        paragraph_mask,
        "Paragraph Mask - " + image_name,
        "gray"
    )


    #Final detected paragraphs
    projection_v2.show_image(
        detected_rgb,
        "Detected Paragraphs - " + image_name
    )

    #displays each paragragh (bug fix for sequence)
    for paragraph_number,paragraph_image in paragraph_images:

        #Convert extracted paragraph to RGB.
        paragraph_rgb = cv2.cvtColor(
            paragraph_image,
            cv2.COLOR_BGR2RGB
        )

        #Display extracted paragraph.
        projection_v2.show_image(
            paragraph_rgb,
            "Paragraph "
            + str(paragraph_number)
            + " - "
            + image_name
        )

###final output section
    print("Number of detected paragraphs:", len(sorted_stats))
    print(image_name, "processing complete.")

#process_image("001")
#process_image("002")
#process_image("003")
#process_image("004")
#process_image("005")
#process_image("006")
#process_image("007")
process_image("008")
print("\nAll Task B images have been processed.")