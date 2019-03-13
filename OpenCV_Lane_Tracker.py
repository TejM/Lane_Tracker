import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

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
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 255, 0),5) 
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = (y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    #print (x1, y1, x2, y2)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines): #SLOPE IS OPPOSITE as Y axis is inverted!!!
    left_side = [] #This is an list of lines with negative slope (ones on the left)
    right_side = [] #This is a n list of lines with positive slope (ones on the right)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1] # intercept is y intercept
        if slope < 0:
            left_side.append((slope, intercept))
        else: 
            right_side.append((slope, intercept))
    #print (left_side)
    #print (right_side)
    left_side_avg = np.average(left_side, axis=0)
    right_side_avg = np.average(right_side, axis=0)
    #print (left_side_avg, 'left side')
    #print (right_side_avg, 'right side')
    left_line = make_coordinates(image, left_side_avg)
    right_line = make_coordinates(image, right_side_avg)
    return np.array([left_line, right_line])
    
for root, dirs, files in os.walk("."):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(filename)
            img_copy = np.copy(img)
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) 
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            canny_image = cv2.Canny(blur, 150, 150)
            cropped_image = roi(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 15, minLineLength = 40, maxLineGap=5 )
            try:
                averaged_lines = average_slope_intercept(img_copy, lines)
                line_image = display_lanes(img_copy, averaged_lines)
            except:
                line_image = display_lanes(img_copy, lines)
            #plt.imshow(img)
            #plt.show()
            combo_image = cv2.addWeighted(img_copy, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            cv2.waitKey(0)
