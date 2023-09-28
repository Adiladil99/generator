import cv2
import numpy as np

def filter2(image, intensity=0.6, saturation=0.7, light_adapt=0.7):
    img = np.float32(image) / 255.0
    mapped_image = np.clip(img * (1.0 + (img * intensity) / (img + 0.01)), 0, 1)
    mapped_image = np.uint8(mapped_image * 255)
    hsv_mapped_image = cv2.cvtColor(mapped_image, cv2.COLOR_BGR2HSV)
    hsv_mapped_image[..., 1] = np.clip(hsv_mapped_image[..., 1] * saturation, 0, 255)
    final_image = cv2.cvtColor(hsv_mapped_image, cv2.COLOR_HSV2BGR)
    final_image = np.clip(final_image * light_adapt, 0, 255).astype(np.uint8)
    return final_image


