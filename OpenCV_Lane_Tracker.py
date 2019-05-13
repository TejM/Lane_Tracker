import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2
import math

"""def canny(imgg):
    gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 150)
    return canny """


def roi(image):
    height, width = image.shape[:2]
    polygons = np.array(
        [[(0, height), (width, height), (width, height*2/3), (0, height*2/3)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lanes(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 5)
    return line_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = (y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    # print (x1, y1, x2, y2)
    return np.array([x1, y1, x2, y2])


def calculate_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
# SLOPE IS OPPOSITE as Y axis is inverted!!!


"""
def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([100, 100, 100])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res"""


def average_slope_intercept(image, lines):
    height, width = image.shape[:2]
    x_mid = width/2
    y_mid = height
    left_side = []
    right_side = []
    min_dist_left = 10000000000
    min_dist_right = 10000000000
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        x_intercept = ((y_mid - y1)/slope) + x1
        if x_intercept < x_mid:
            distL = calculate_distance(x_mid, y_mid, x_intercept, y_mid)
            if distL < min_dist_left:
                min_dist_left = distL
                left_side = [(slope, intercept)]
        else:
            distR = calculate_distance(x_mid, y_mid, x_intercept, y_mid)
            if distR < min_dist_right:
                min_dist_right = distR
                right_side = [(slope, intercept)]

    left_side_avg = np.average(left_side, axis=0)
    right_side_avg = np.average(right_side, axis=0)
    left_line = make_coordinates(image, left_side_avg)
    right_line = make_coordinates(image, right_side_avg)
    return np.array([left_line, right_line])


for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".png"):
            img = cv2.imread(filename)
            img_copy = np.copy(img)
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            cropped = roi(blur)
            canny_image = cv2.Canny(cropped, 100, 200)
            #cv2.imshow("result", canny_image)
            # cv2.waitKey(0)
            lines = cv2.HoughLinesP(
                canny_image, 1, np.pi/180, 4, minLineLength=30, maxLineGap=10)

            try:
                averaged_lines = average_slope_intercept(img_copy, lines)
                line_image = display_lanes(img_copy, averaged_lines)
            except:
                line_image = display_lanes(img_copy, lines)

            combo_image = cv2.addWeighted(img_copy, 0.8, line_image, 1, 1)
            #cv2.imshow("result", combo_image)
            # cv2.waitKey(0)
            cv2.imwrite(filename + "_"+".png", combo_image)
            #averaged_lines = average_slope_intercept(img_copy, lines)
