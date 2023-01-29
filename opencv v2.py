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



capture = cv2.VideoCapture('Test videos/LA Walk Park.mp4')
while capture.isOpened():
    _, frame = capture.read()
    cv2.imshow('Frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gaussian = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gray,5)

    dst = cv2.Canny(median, 50, 200, None, 3)
    #  Standard Hough Line Transform
    lines = cv2.HoughLines(dst, 1, np.pi/180, 150, None, 0, 0)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if (abs(pt1[1]-pt2[1]) > 70) and (find_angle(pt1, pt2) > 30): # filter line height and angle
                cv2.line(frame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
    cv2.imshow("Lines", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()