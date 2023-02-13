import cv2
import numpy as np
import math

def find_line_angle(pt1, pt2):

    '''Find angle of a line with respect to horizontal
    given two points the line passes through'''
    
    x1, y1 = pt1
    x2, y2 = pt2
    x = abs(x1 - x2)
    y = y1 - y2
    if x == 0:
        x = 0.1
    angle = math.degrees(math.atan(y/x))
    angle = round(angle, 3)
    return angle

def display_view_angle(intersection_point):

    '''Show the camera's vertical and horizontal
    direction of viewing with respect to intersection
    point(vanishing point) of perspective lines'''
    
    if intersection_point == None:
        return

    x_direction = None
    y_direction = None
    x, y = intersection_point

    if x <= (img_width * 0.3):
        x_direction = 'Right >'
    elif x >= (img_width * 0.7):
        x_direction = 'Left <'
    else:
        x_direction = 'Centre ^'

    if y <= (img_height * 0.3):
        y_direction = 'Up'
    elif y >= (img_height * 0.7):
        y_direction = 'Down'
    else:
        y_direction = 'Level'

    result_text = x_direction + ',' + y_direction
    cv2.putText(frame, result_text, (int(img_width * 0.1), int(img_height * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    return

class intersection:
    prev_intersection = None
    validation_weight = 5

    def find_intersection(pt1, pt2, pt3, pt4):

        '''Calculate the intersection point of two lines
        given two points from each of the two line'''

        x1, y1 = pt1
        x2, y2 = pt2
        x3, y3 = pt3
        x4, y4 = pt4

        a = ((x1 - x3) * (y2 - y3)) - ((x2 - x3) * (y1 - y3))
        b = ((x1 - x4) * (y2 - y4)) - ((x2 - x4) * (y1 - y4))

        try:
            x = int(x3 + ((a/(a-b)) * (x4 - x3)))
            y = int(y3 + ((a/(a-b)) * (y4 - y3)))

            if x < 0:
                x = 0
            elif x > img_width:
                x = img_width
            if y < 0:
                y = 0
            elif y > img_height:
                y = img_height

            #print(x,y)
        except ZeroDivisionError:
            return
        return (x, y)

    def validate_point(intersect_point , pixel_gap):

        '''Function to check the consistency of intersection points
        with respect to the previous point coordinates and updates
        the intersection point only when it passes a certain vote count'''

        # for first frame
        if intersection.prev_intersection == None or intersect_point == None:
            intersection.prev_intersection = intersect_point
            return intersect_point

        # for adjusting weight with respect to pixel gap
        if (abs(intersect_point[0] - intersection.prev_intersection[0]) > pixel_gap) or (abs(intersect_point[1] - intersection.prev_intersection[1]) > pixel_gap):
            intersection.validation_weight = intersection.validation_weight - 5
        else:
            intersection.validation_weight = intersection.validation_weight + 1

        # for limiting weight
        if intersection.validation_weight > 10:
            intersection.validation_weight = 10
        
        # if weight < 0, reset weight and use the previous point as current point
        if intersection.validation_weight <= 0:
            intersection.validation_weight = 5
            return intersection.prev_intersection
        
        # if everything is good, return the original point
        intersection.prev_intersection = intersect_point
        return intersect_point    
            
capture = cv2.VideoCapture('Test videos/LA Walk Park.mp4')

img_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
img_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while capture.isOpened():
    _, frame = capture.read()

    cv2.imshow('Frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,3)
    dst = cv2.Canny(median, 50, 200, None, 3)
    #  Standard Hough Line Transform
    lines = cv2.HoughLines(dst, 1, np.pi/180, 150, None, 0, 0)
    perspective_lines = []
    barrier_lines = []

    if lines is not None:
        for i in range(0, min(7, len(lines))):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            line_angle = find_line_angle(pt1, pt2)
            
            # filter lines
            if (max(pt1[1],pt2[1]) > (img_height * 0.6)) and (abs(line_angle) < 40) and (len(perspective_lines) <= 2):
                perspective_lines.append([pt1, pt2])
            if (line_angle > 80):
                print(line_angle)
                barrier_lines.append([pt1, pt2])
            
    if barrier_lines != []:
        for i in range(0, len(barrier_lines)):
            cv2.line(frame, pt1, pt2, (200,200,0), 2, cv2.LINE_AA) # Draw the lines

    
    if perspective_lines is not None:
        for i in range(len(perspective_lines)):
            cv2.line(frame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA) # Draw the lines
            try:
                intersect_point = intersection.find_intersection(perspective_lines[i][0], perspective_lines[i][1], perspective_lines[i-1][0], perspective_lines[i-1][1])
                validated_point = intersection.validate_point(intersect_point, 50)
                cv2.circle(frame, validated_point, 4, (0, 255, 0), -1)
                display_view_angle(validated_point)
            except IndexError:
                print('Error')

    cv2.imshow("Lines", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()