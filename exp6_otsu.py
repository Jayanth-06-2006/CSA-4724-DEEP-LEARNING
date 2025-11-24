
"""Experiment 6: Otsu's thresholding for a grayscale image.
Input: 'input_gray.jpg' (place grayscale image in same folder or supply color image)
Output: binary image saved as 'exp6_otsu_binary.png' and threshold printed.
"""
import cv2
import numpy as np
import sys

img = cv2.imread("input_gray.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File input_gray.jpg not found in working directory. Please add the image and re-run.")
else:
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Computed Otsu threshold:", ret)
    cv2.imwrite("exp6_otsu_binary.png", th)
    print("Saved binary image to exp6_otsu_binary.png")
