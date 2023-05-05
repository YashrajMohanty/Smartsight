import cv2
from ultralytics import YOLO
import numpy as np

class obj_detect():

    def __init__(self, weights):
        self.boxes = []
        self.cls = []
        print('Initializing model:', weights.split('/')[-1])
        # "Models/best.pt" #custom model
        # "ultralytics/yolov8n.pt" #default model
        self.model = YOLO(weights)
        print('Complete')


    def boundingboxcenter(self,frame):
        boxes = self.boxes
        if len(boxes) == 0:
            return

        bb_center = []

        for box in boxes:
            avg_x = int((box[0] + box[2])/2)
            avg_y = int((box[1] + box[3])/2)
            bb_center.append([avg_x, avg_y])
            cv2.circle(frame, (avg_x, avg_y), 2, (0, 255, 0), -1)
        return bb_center


    def filter_classes(self, result):
        filter_class = [0,1,2,3,5,7,13,15,16,17,18,19,57] # classes to filter (according to COCO128)
        result_boxes = result.boxes.xyxy.cpu().numpy()
        result_cls = result.boxes.cls.cpu().numpy()     
        boxes = []
        cls = []
        flag = False
        for i in range(len(result_cls)):
            for j in filter_class:
                if result_cls[i] == j:
                    flag = True
                    continue
            if flag:
                cls.append(result_cls[i])
                boxes.append(result_boxes[i])
        cls = np.array(cls)
        boxes = np.array(boxes)
        return (boxes, cls)


    def detect_objects(self, frame, filter_class=False):
        results = self.model.predict(source=frame, verbose=False)
        if len(results) > 0:
            results_plot = results[0].plot(show_conf=False, line_width=1)
            for result in results:
                if filter_class: # if using default model, filter classes
                    self.boxes, self.cls = self.filter_classes(result)
                else:
                    self.boxes = result.boxes.xyxy.cpu().numpy()
                    self.cls = result.boxes.cls.cpu().numpy()
        return results_plot


if __name__ == "__main__":

    obj_det = obj_detect("ultralytics/yolov8n.pt")
    capture = cv2.VideoCapture('Test videos/LA Walk Park.mp4')
    #capture = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    print('Engaging test')
    while capture.isOpened():
        _, frame = capture.read()

        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break
        
        results_plot = obj_det.detect_objects(frame, filter_class=True)
        bb_center = obj_det.boundingboxcenter(results_plot)
        cv2.imshow("YOLOv8", results_plot)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()