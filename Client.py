import cv2
import socket
import pickle
import struct

client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
try:
    port = 8089
    client_socket.connect(('localhost', port))
    print('Connected to [server address] on port', port)
except socket.error:
    print('Connection to [server address] on port', port, 'failed')
    quit()

captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')

while True:
    _, frameL = captureL.read()
    _, frameR = captureR.read()

    if (str(type(frameL))) == "<class 'NoneType'>":
        print('Stream ended')
        break

    frame = cv2.hconcat([frameL,frameR])
    # Serialize frame

    data = pickle.dumps(frame)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED
    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break
    # Then data
    try:
        client_socket.sendall(message_size + data)
    except ConnectionResetError:
        print('Connection closed')
        break

client_socket.shutdown(1)
client_socket.close()
captureL.release()
captureR.release()
cv2.destroyAllWindows()