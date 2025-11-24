
"""Experiment 7: Watershed algorithm to separate overlapping objects.
Input: 'overlap.jpg' (image with overlapping circular/round objects)
Output: 'exp7_watershed_segments.png' showing separated markers.
"""
import cv2
import numpy as np
import sys

img = cv2.imread("overlap.jpg")
if img is None:
    print("File overlap.jpg not found in working directory. Please add the image and re-run.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0,0,255]  # boundary in red
    cv2.imwrite("exp7_watershed_segments.png", img)
    print("Saved segmented image to exp7_watershed_segments.png")
