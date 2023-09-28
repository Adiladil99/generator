import os
import cv2

def filter6(originalmage):
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C,   cv2.THRESH_BINARY, 9, 9)
    colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    return cartoonImage