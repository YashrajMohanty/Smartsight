if __name__ == "__main__":
    
    import cv2
    import Obj_detect_v8
    import Stereo_video
    import Audio_feedback as af
    
    obj_det = Obj_detect_v8.obj_detect("ultralytics/yolov8n.pt")
    stereo = Stereo_video.stereo_cam()
    stereo.calibrate_stereo()
    af.alert_system.start_play_thread()

    captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')
    
    while captureL.isOpened():

        _, frameL = captureL.read()
        _, frameR = captureR.read()
        
        if (str(type(frameL))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        results_plot = obj_det.detect_objects(frameL,filter_class=True)
        bb_center = obj_det.boundingboxcenter(results_plot)

        disp_map = stereo.run_stereo(frameL, frameR)
        stereo.place_markers(disp_map)

        distances = stereo.find_distance(results_plot, bb_center, False)
        cls = obj_det.cls
        obs_flag = stereo.obstruction_flag

        af.alert_system.check(cls, distances, obs_flag)
        cv2.imshow("YOLOv8", results_plot)
        cv2.imshow("Stereo", disp_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()