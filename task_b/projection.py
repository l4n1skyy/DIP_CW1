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