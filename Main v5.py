import cv2
import numpy as np
import torch

#os.system('C:/Users/yashr/Desktop/CVproject/YOLOv7/detect.py --weights C:/Users/yashr/Desktop/CVproject/YOLOv7/yolov7.pt --source 0 --device 0 --img-size 768 --no-trace --exist-ok')
#model = attempt_load('YOLOv7/yolov7.pt', map_location='cuda:0')  # load FP32 model

model = torch.hub.load("YOLOv5", 'custom', path="YOLOv5/yolov5s.pt", source='local')

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    results = model(frame)
    cv2.imshow('YOLOv5', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()