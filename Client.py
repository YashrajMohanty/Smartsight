import cv2
import socket
import struct
from io import BytesIO
import numpy as np

import Audio_feedback as af
af.alert_system.alerts = []
af.alert_system.start_play_thread()

#----------------------------CLIENT---------------------------
client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
payload_size = struct.calcsize("L")
try:
    port = 8089
    client_socket.connect(('localhost', port))
    print('Connected to localhost on port', port)
except socket.error:
    print('Connection to localhost on port', port, 'failed!')
    quit()
#---------------------------/CLIENT---------------------------

def pack_frame(frameL, frameR):
    img_file = BytesIO()
    frame = cv2.hconcat([frameL,frameR])
    # Serialize frame
    np.save(img_file, frame)
    data = img_file.getvalue()
    # Send message length
    message_size = struct.pack("L", len(data))
    return message_size + data

def get_alerts():
    # Retrieve message size
    data = b''
    
    while len(data) < payload_size:
        data += client_socket.recv(1024)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += client_socket.recv(1024*32) # 32kb buffer size

    alert_data = data[:msg_size]
    data = data[msg_size:]

    alerts = alert_data.decode("utf-8")
    alerts = alerts.split("$")
    return alerts


captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')

while True:
    _, frameL = captureL.read()
    _, frameR = captureR.read()

    if (str(type(frameL))) == "<class 'NoneType'>":
        print('Stream ended')
        break

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
    
    try:
        client_socket.sendall(pack_frame(frameL, frameR))
    except ConnectionResetError:
        print('Connection closed')
        break

    af.alert_system.alerts = get_alerts()

captureL.release()
captureR.release()
client_socket.close()