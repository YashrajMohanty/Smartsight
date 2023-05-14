import socket
import struct
import cv2
from io import BytesIO
import numpy as np

import Obj_detect_v8
import Stereo_video
import Audio_feedback as af

#--------------------------SERVER------------------------
HOST = ''
PORT = 8089
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
server_socket.bind((HOST, PORT))
print('Socket bind complete')
server_socket.listen(10)
print('Socket now listening')
conn, addr = server_socket.accept()
payload_size = struct.calcsize("L")
#-------------------------/SERVER-------------------------

obj_det = Obj_detect_v8.obj_detect("ultralytics/yolov8n.pt")
stereo = Stereo_video.stereo_cam()
stereo.calibrate_stereo()
#af.alert_system.start_play_thread()

def get_frame():
    # Retrieve message size
    data = b''
    img_file = BytesIO()
    
    while len(data) < payload_size:
        data += conn.recv(1024)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096*8) # 32kb buffer size

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    img_file.write(frame_data)
    img_file.seek(0)
    frame = np.load(img_file)
    return frame

def pack_alerts(alerts):
    alerts = ("$").join(alerts)
    data = alerts.encode("utf-8")
    message_size = struct.pack("L", len(data))
    return message_size + data
   

while True:

    #-----------------------------SERVER--------------------------
    frame = get_frame()

    if (str(type(frame))) == "<class 'NoneType'>":
        print('Stream ended')
        break

    frameL = frame[:, :int(frame.shape[1]/2)]
    frameR = frame[:, int(frame.shape[1]/2):]
    #----------------------------/SERVER-------------------------

    results_plot = obj_det.detect_objects(frameL,filter_class=True)
    bb_center = obj_det.boundingboxcenter(results_plot)

    disp_map = stereo.run_stereo(frameL, frameR)
    stereo.place_markers(disp_map)

    distances = stereo.find_distance(results_plot, bb_center, False)
    cls = obj_det.cls
    obs_flag = stereo.obstruction_flag

    af.alert_system.check(cls, distances, obs_flag)
    alerts = af.alert_system.alerts
    alerts = pack_alerts(alerts)
    conn.sendall(alerts)

    cv2.imshow("YOLOv8", results_plot)
    cv2.imshow("Stereo", disp_map)

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
cv2.destroyAllWindows()
server_socket.close()