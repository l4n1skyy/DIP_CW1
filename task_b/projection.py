# This file comtains the functions 
# used for histogram projections ,
# so the main.py folder is more efficient.

import numpy as np
import matplotlib.pyplot as plt

#horizontal projection
def horizontal_projection(binary_image):
    horizontal = np.count_nonzero(binary_image == 0,axis=1)
    return horizontal

#vertical projection
def vertical_projection(binary_image):
    vertical = np.count_nonzero(binary_image ==0,axis=0)
    return vertical

#find text ranges
def find_projection_ranges(projection_values):
    active =(projection_values > 0).astype(np.uint8) #converts projection into active & inactive positions

    padded = np.pad(active, (1, 1), constant_values = 0) #padding for before and after projection

    difference = np.diff(padded) #finds changes between active and inactive positions

    start_positions = np.where(difference == 1)[0] #change from 0 --> 1 represents starting position
    end_positions = np.where(difference == -1)[0] -1 #change from 0 --> represents ending position

    return start_positions, end_positions

#Column split shenanigans
def find_column_split(vertical_values):
    
    page_width = len(vertical_values) #getting the width of the projection

    #search the center of the page
    centre_start = int(page_width * 0.40)
    centre_end = int(page_width * 0.60)

    #finding the position with the lowest amount of pixel
    column_split = centre_start + np.argmin(vertical_values[centre_start:centre_end])

    return column_split

#Sorting Functions
def sort_paragraphs(paragraph_stats, column_split):

    x_position = paragraph_stats[:,0]
    y_position = paragraph_stats[:,1]

    paragraph_width = paragraph_stats[:,2]

    #calculate for the center of paragrapghs
    x_centre = x_position + (paragraph_width / 2)


    column_index = (x_centre >= column_split).astype(np.int32)

    order = np.lexsort((y_position,column_index))

    #for arranging the paragraphs
    sorted_stats = paragraph_stats[order]

    return sorted_stats

#projection display
def show_projection(projection_values, title, axis_name):

    plt.figure(figsize=(10, 4))

    plt.plot(projection_values)

    plt.title(title)

    plt.xlabel(axis_name)

    plt.ylabel("Number of Black Text Pixels")

    plt.tight_layout()

    plt.show()