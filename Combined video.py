if __name__ == "__main__":
    
    import cv2
    import Obj_detect_v8 as obj
    import Stereo_video as sv
    import Audio_feedback as af

    sv.stereo_cam.calibrate_stereo()
    af.alert_system.start_play_thread()

    captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')
    
    while captureL.isOpened():

        _, frameL = captureL.read()
        _, frameR = captureR.read()
        

        if (str(type(frameL))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        results_plot = obj.obj_detect.detect_objects(frameL,filter_class=True)
        bb_center = obj.obj_detect.boundingboxcenter(results_plot)

        disp_map = sv.stereo_cam.run_stereo(frameL, frameR)
        sv.stereo_cam.place_markers(disp_map)

        distances = sv.stereo_cam.find_distance(results_plot, bb_center, True)
        cls = obj.obj_detect.cls
        obs_flag = sv.stereo_cam.obstruction_flag

        af.alert_system.check(cls, bb_center, distances, obs_flag)
        cv2.imshow("YOLOv8", results_plot)
        cv2.imshow("Stereo", disp_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()