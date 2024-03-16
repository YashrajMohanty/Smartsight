from threading import Thread, Lock
from winsound import Beep
from time import sleep

thread_lock = Lock()
Lfreq = 500
Rfreq = 500

def set_freq(left, right):
    global Lfreq
    global Rfreq
    with thread_lock:
        Lfreq = int((left/64) * 600) + 400 # 400-1000Hz
        Rfreq = int((right/64) * 600) + 400

def audio_loop():
    while(1):
        Beep(Lfreq,100)
        Beep(Rfreq,100)
        sleep(1)

audio_thread = Thread(target=audio_loop, args=(), name='AudioThread')
audio_thread.daemon = True

def start_thread():
    audio_thread.start()
    print("Audio thread started")

if __name__ == "__main__":
    audio_loop()