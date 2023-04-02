import cv2
from ultralytics import YOLO

print('Initializing model')
#model = YOLO("ultralytics/yolov8n.pt") #default model
model = YOLO("Custom model/best.pt") #custom model
print('Engaging...')


class obj_detect():
    boxes = []
    cls = []

    def boundingboxcenter(frame):
        boxes = obj_detect.boxes
        if len(boxes) == 0:
            return

        bb_center = []

        for box in boxes:
            avg_x = int((box[0] + box[2])/2)
            avg_y = int((box[1] + box[3])/2)
            bb_center.append([avg_x, avg_y])
            cv2.circle(frame, (avg_x, avg_y), 2, (0, 255, 0), -1)
        return bb_center

    def detect_objects(frame):
        results = model.predict(source=frame, verbose=False)
        if len(results) > 0:
            results_plot = results[0].plot(show_conf=False, line_width=1)
            for r in results:   
                obj_detect.boxes = r.boxes.xyxy.cpu().numpy()
                obj_detect.cls = r.boxes.cls.cpu().numpy()

        return results_plot


if __name__ == "__main__":

    capture = cv2.VideoCapture('Test videos/LA Walk Park.mp4')
    print('Engaging test')
    while capture.isOpened():
        _, frame = capture.read()

        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        results_plot = obj_detect.detect_objects(frame)
        bb_center = obj_detect.boundingboxcenter(results_plot)
        cv2.imshow("YOLOv8", results_plot)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()