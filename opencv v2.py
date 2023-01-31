import cv2
import numpy as np
import math

def find_angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    angle = math.degrees(math.atan(x/y))
    return angle

def find_intersection(pt1, pt2, pt3, pt4):
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

        print(x,y)
        cv2.circle(frame, (x,y), 4, (255, 0, 0), -1)
    except ZeroDivisionError:
        return

    return (x, y)

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
    lines_filtered = []

    if lines is not None:
        for i in range(0, min(2,len(lines))):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if (max(pt1[1],pt2[1]) > (img_height * 0.6)) and (find_angle(pt1, pt2) > 40): # Filter line height and angle
                lines_filtered.append([pt1, pt2])
    
    if lines_filtered is not None:
        for i in range(0, len(lines_filtered)):
            cv2.line(frame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA) # Draw the lines
            try:
                intersection = find_intersection(lines_filtered[i][0], lines_filtered[i][1], lines_filtered[i-1][0], lines_filtered[i-1][1])
            except IndexError:
                print('Error')

    cv2.imshow("Lines", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()