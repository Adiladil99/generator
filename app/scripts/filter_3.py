import numpy as np
import cv2

def filter3(image):
    num_colors = 8
    step_size = 256 // num_colors
    posterized_image = (image // step_size) * step_size
    return posterized_image
