from ultralytics import YOLO
import torch

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
        
        x_avg = torch.add(boxes[:,0], boxes[:,2])
        y_avg = torch.add(boxes[:,1], boxes[:,3])

        bb_center = torch.stack((x_avg, y_avg), dim=1)
        bb_center = torch.div(bb_center, 2, rounding_mode="trunc")

        return bb_center


    def __filter_classes(self, result):
        filter_class = [0,1,2,3,5,7,13,15,16,17,18,19,57] # classes to filter in (according to COCO128)
        cls = result.boxes.cls.int() #[x1, y1, x2, y2]
        boxes = result.boxes.xyxy.int()

        new_cls = torch.tensor([], dtype=torch.int32, device=torch.device('cuda'))
        new_boxes = torch.tensor([], dtype=torch.int32, device=torch.device('cuda'))

        for i in range(len(cls)):
            for j in filter_class:
                if cls[i] == j:
                    new_cls = torch.cat((new_cls, cls[i].unsqueeze(-1)),dim=0)
                    new_boxes = torch.cat((new_boxes, boxes[i,:].unsqueeze(0)), dim=0)
                    break

        return (new_boxes, new_cls)


    def detect_objects(self, frame, filter_class=False):
        results = self.model.predict(source=frame, verbose=False)
        if len(results):
            results_plot = results[0].plot(show_conf=False, line_width=1)
            for result in results:
                if filter_class: # if using default model, filter classes
                    self.boxes, self.cls = self.__filter_classes(result)
                else:
                    self.boxes = result.boxes.xyxy
                    self.cls = result.boxes.cls
        return results_plot


if __name__ == "__main__":

    import cv2

    obj_det = obj_detect("ultralytics/yolov8n.pt")
    capture = cv2.VideoCapture('Chessboard/LA Walk Park.mp4')
    #capture = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    print('Engaging test')
    while capture.isOpened():
        _, frame = capture.read()

        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break
        
        results_plot = obj_det.detect_objects(frame, filter_class=True)
        bb_center = obj_det.boundingboxcenter(results_plot)

        if bb_center != None:
            for i in range(len(bb_center)):
                cv2.circle(results_plot, bb_center[i,:].tolist(), 2, (0, 255, 0), -1)

        cv2.imshow("YOLOv8", results_plot)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()