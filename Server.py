import socket
import struct
import cv2
from io import BytesIO
import numpy as np
from time import time

import Obj_detect_v8
import Stereo_CUDA
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
midas = Stereo_CUDA.stereo_midas()
sd = Stereo_CUDA.stereo_dist()

prev_frame_time = 0
new_frame_time = 0

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
        data += conn.recv(1024*256) # 256kb buffer size

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
    #----------------------------/SERVER-------------------------

    results_plot = obj_det.detect_objects(frame,filter_class=True)
    bb_center = obj_det.boundingboxcenter(results_plot)

    disp_map = midas.predict_depth(frame)
    sd.place_markers(disp_map)

    distances = sd.find_distance(disp_map, bb_center, True, 0.5)
    cls = obj_det.cls

    af.alert_system.check(cls, distances, sd.obstruction_flag, sd.caution_flag)
    alerts = af.alert_system.alerts
    alerts = pack_alerts(alerts)
    conn.sendall(alerts)

    new_frame_time = time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(results_plot, str(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

    cv2.imshow("YOLOv8", results_plot)
    cv2.imshow("Stereo", midas.convert_to_thermal(disp_map))

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
cv2.destroyAllWindows()
server_socket.close()