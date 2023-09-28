import cv2

def filter5(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return equalized_color
