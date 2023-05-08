import pickle
import socket
import struct
import cv2

import Obj_detect_v8
import Stereo_video
import Audio_feedback as af

HOST = ''
PORT = 8089
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
server_socket.bind((HOST, PORT))
print('Socket bind complete')
server_socket.listen(10)
print('Socket now listening')
conn, addr = server_socket.accept()
data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED
#------------------------------------------------------

obj_det = Obj_detect_v8.obj_detect("ultralytics/yolov8n.pt")
stereo = Stereo_video.stereo_cam()
stereo.calibrate_stereo()
af.alert_system.start_play_thread()

while True:

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    frame = pickle.loads(frame_data)

    if (str(type(frame))) == "<class 'NoneType'>":
        print('Stream ended')
        break

    frameL = frame[:, :int(frame.shape[1]/2)]
    frameR = frame[:, int(frame.shape[1]/2):]

    #-------------------------------------------------

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