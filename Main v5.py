import cv2
import numpy as np
import torch

model = torch.hub.load("YOLOv5", 'custom', path="YOLOv5/yolov5s.pt", source='local')

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()

    #--- YOLOv5 ---
    results = model(frame)
    results_pd = results.pandas().xyxy[0]
    try: #error handling in case number of records is zero
        print(results_pd['name'][0])
    except:
        pass
    #--- YOLOv5 ---
    
    cv2.imshow('YOLOv5', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()