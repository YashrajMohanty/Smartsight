import cv2
import numpy as np
import torch
import pandas

model = torch.hub.load("YOLOv5", 'custom', path="YOLOv5/yolov5s.pt", source='local')

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()

    _,_,red = cv2.split(frame) # taking red channel from BGR image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # BGR to HSV
    _,saturation,value = cv2.split(hsv) # taking saturation and value(lightness) channels from HSV image
    sobel_x = cv2.Sobel(value, cv2.CV_64F, 1, 0, 3) # sobel edge detection on x-axis using value channel
    sobel_y = cv2.Sobel(value, cv2.CV_64F, 0, 1, 3) # sobel edge detection on y-axis using value channel
    _, thresh_s = cv2.threshold(saturation, 80, 255, cv2.THRESH_BINARY) # binary thresholding on saturation channel
    _, thresh_r = cv2.threshold(red, 125, 255, cv2.THRESH_BINARY) # binary thresholding on red channel
    #adaptive_thresh = cv2.adaptiveThreshold(saturation, 225, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 12)
    canny = cv2.Canny(frame, 125, 175) #edge cascade

    #--- YOLOv5 ---
    results = model(frame)
    results_pd = results.pandas().xyxy[0]
    try: #error handling in case number of records is zero
        print(results_pd['name'][0])
    except:
        pass
    #--- YOLOv5 ---
    
    cv2.imshow('YOLOv5', np.squeeze(results.render()))
    #cv2.imshow('Thresh S', thresh_s)
    #cv2.imshow('Thresh R', thresh_r)
    #cv2.imshow('Sobel Y', sobel_y)
    #cv2.imshow('Sobel X', sobel_x)
    #cv2.imshow('Adapthresh', adaptive_thresh)
    #cv2.imshow('canny', canny)

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()