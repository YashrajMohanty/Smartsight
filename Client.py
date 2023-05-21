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

def pack_frame(frame):
    img_file = BytesIO()
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
        try:
            data += client_socket.recv(1024)
        except ConnectionResetError:
            print("Connection closed")
            client_socket.close()
            quit()

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


capture = cv2.VideoCapture("Chessboard/LA Walk Park.mp4")

while True:
    _, frame = capture.read()

    if (str(type(frame))) == "<class 'NoneType'>":
        print('Stream ended')
        break

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
    
    try:
        client_socket.sendall(pack_frame(frame))
    except ConnectionResetError:
        print('Connection closed')
        break

    af.alert_system.alerts = get_alerts()

    cv2.imshow('Response',np.zeros((60,1)))
    cv2.setWindowProperty('Response', cv2.WND_PROP_TOPMOST,1)
    if cv2.waitKey(10) & 0xFF == ord('a'): #press a to toggle alerts
        af.alert_system.toggle_speech()
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
client_socket.close()