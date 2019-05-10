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
    polygons = np.array([[(0, 145), (140, 130), (125, 80), (5, 80)]])
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


def average_slope_intercept(image, lines):
    height, width = image.shape[:2]
    x_mid = width/2
    # This is an list of lines with negative slope (ones on the left)

    # This is a n list of lines with positive slope (ones on the right)

    min_dist_left = 1000
    min_dist_right = 1000
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]  # intercept is y intercept
        if x1 < x_mid:
            cv2.circle(image, (x1, y1), 6, (0, 0, 255), -1)
            #cv2.imshow("result", image)
            # cv2.waitKey(0)
            distL = calculate_distance(x_mid, height, x1, y1)
            if distL < min_dist_left:
                min_dist_left = distL
                left_side = [(slope, intercept)]
            # closest leftline"""
            # left_side.append((slope, intercept))
        else:
            # right_side.append(slope, intercept)
            cv2.circle(image, (x2, y2), 6, (0, 0, 255), -1)
            #cv2.imshow("result", image)
            # cv2.waitKey(0)
            distR = calculate_distance(x_mid, height, x2, y2)
            if distR < min_dist_right:
                min_dist_right = distR
                right_side = [(slope, intercept)]
            # closest leftline
    # print (left_side)
    # print (right_side)
    left_side_avg = np.average(left_side, axis=0)
    right_side_avg = np.average(right_side, axis=0)
    left_line = make_coordinates(image, left_side_avg)
    right_line = make_coordinates(image, right_side_avg)
    print (left_line, 'left line')
    print (right_line, 'right line')
    return np.array([left_line, right_line])


for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(filename)
            img_copy = np.copy(img)
            # gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(img_copy, (5, 5), 0)
            canny_image = cv2.Canny(img_copy, 150, 150)
            cropped_image = roi(canny_image)
            lines = cv2.HoughLinesP(
                canny_image, 1, np.pi/180, 15, minLineLength=40, maxLineGap=7)

            averaged_lines = average_slope_intercept(img_copy, lines)
            line_image = display_lanes(img_copy, averaged_lines)
            # plt.imshow(img)
            # plt.show()
            combo_image = cv2.addWeighted(img_copy, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            print(filename)
            cv2.waitKey(0)
