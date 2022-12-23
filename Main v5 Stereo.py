import cv2
import numpy as np
import torch

#os.system('python C:/Users/yashr/Desktop/GIT/Smartsight/YOLOv5/detect.py --weights C:/Users/yashr/Desktop/GIT/Smartsight/YOLOv5/yolov5s.pt --source C:/Users/yashr/Desktop/GIT/Smartsight/"Test videos"/"LA Walk "Outdoor.mp4')

model = torch.hub.load("YOLOv5", 'custom', path="YOLOv5/yolov5s.pt", source='local')

def measure(img, x, y, centre_x, centre_y):
    cv2.line(img, (centre_x, centre_y), (x,y), (0,0,255), 2)
    dist = (((x - centre_x)**2) + ((y - centre_y)**2))**(1/2)
    cv2.putText(img, str(int(dist)), (centre_x - 40, centre_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return

capture_left = cv2.VideoCapture('Test videos/Stereo L.mp4')
capture_right = cv2.VideoCapture('Test videos/Stereo R.mp4')

while capture_left.isOpened():
    _, frame_l = capture_left.read()
    _, frame_r = capture_right.read()

    #--- YOLOv5 ---
    results_l = model(frame_l)
    results_r = model(frame_r)

    results_l_pd = results_l.pandas().xyxy[0]
    results_r_pd = results_r.pandas().xyxy[0]

    #print(int(results_l_pd['xmin'][0]))

    results_l_img = np.squeeze(results_l.render())
    results_r_img = np.squeeze(results_r.render())

    try:
        measure(results_l_img, int(results_l_pd['xmin'][0]), int(results_l_pd['ymin'][0]), 320, 480)
        measure(results_r_img, int(results_r_pd['xmin'][0]), int(results_r_pd['ymin'][0]), 320, 480)
    except:
        pass
    #--- YOLOv5 ---
    
    cv2.imshow('YOLOv5 L', results_l_img)
    cv2.imshow('YOLOv5 R', results_r_img)

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture_left.release()
capture_right.release()
cv2.destroyAllWindows()