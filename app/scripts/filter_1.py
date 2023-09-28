import cv2
import numpy as np

def filter1(image):
    brightness=0
    contrast=1
    adjusted_image = image.copy()
    adjusted_image = cv2.addWeighted(adjusted_image, contrast, np.zeros(adjusted_image.shape, adjusted_image.dtype), 0, brightness).astype(np.uint8)
    return adjusted_image