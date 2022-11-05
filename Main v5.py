import cv2
import numpy as np
import torch

model = torch.hub.load("YOLOv5", 'custom', path="YOLOv5/yolov5s.pt", source='local')

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #--- YOLOv5 ---
    results = model(frame)
    #--- YOLOv5 ---
    cv2.imshow('YOLOv5', np.squeeze(results.render()))
    cv2.imshow('Thresh', thresh)
    
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()