if __name__ == "__main__":
    
    import cv2
    from time import time
    import Obj_detect_v8
    import Depth_DNN
    import Audio_feedback as af
    
    obj_det = Obj_detect_v8.obj_detect("ultralytics/yolov8n.pt")
    midas = Depth_DNN.midas()
    sd = Depth_DNN.distance_estimation()
    af.alert_system.start_play_thread()

    prev_frame_time = 0
    new_frame_time = 0

    capture = cv2.VideoCapture('Chessboard/LA Walk Park.mp4')

    while capture.isOpened():

        _, frame = capture.read()

        
        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        results_plot = obj_det.detect_objects(frame,filter_class=True)
        bb_center = obj_det.boundingboxcenter(results_plot)

        disp_map = midas.predict_depth(frame)
        sd.place_markers(disp_map)

        distances = sd.find_distance(disp_map, bb_center, True, 0.5)
        cls = obj_det.cls

        af.alert_system.check(cls, distances, sd.obstruction_flag, sd.caution_flag)

        new_frame_time = time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        cv2.putText(results_plot, str(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow("YOLOv8", results_plot)
        cv2.imshow("Stereo", midas.convert_to_thermal(disp_map))
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()