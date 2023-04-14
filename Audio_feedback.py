from time import sleep
from threading import Thread


class alert_system:

    alerts = []
    def check(cls, bb_center, distances, obs_flag=False):

        if len(cls) == 0:
            return

        left_partition = 270 # 320-50
        right_partition = 470 # 320+50

        obj_set = ['vehicle', 'vehicle', 'vehicle', 'animal', 'animal', 'vehicle', 'person', 'train', 'vehicle'] #['bicycle', 'bus', 'car', 'cat', 'dog', 'motorcycle', 'person', 'train', 'truck']
        cls = cls.astype(int)
        cls = list(cls)
        for i in range(len(cls)):
            cls[i] = obj_set[cls[i]]
        
        left = []
        right = []
        center = []
        sections = (center, left, right) 

        for i in range(len(bb_center)):
            if distances[i] > 8: # only objects closer than this distance(meters) will be alerted of
                continue
            if (bb_center[i][0] < left_partition):
                left.append(cls[i])
            if (bb_center[i][0] > right_partition):
                right.append(cls[i])
            else:
                center.append(cls[i])

        alerts_left = []
        alerts_right = []
        alerts_center = []
        alerts = [alerts_center, alerts_left, alerts_right]

        directions = ['front','left','right']

        for i in range(3):
            animal_count = 0
            person_count = 0
            vehicle_count = 0
            train_count = 0
            side = sections[i]
            
            if len(side) == 0:
                continue

            for j in side:
                if len(j) == 0:
                    continue
                if j == 'animal':
                     animal_count += 1
                if j == 'person':
                    person_count += 1

                if j == 'train':
                    train_count += 1

                if j == 'vehicle':
                     vehicle_count += 1

            if obs_flag:
                alert = "Obstruction"
                alerts[i].append(alert)
            if animal_count > 1:
                alert = "Animals to the " + directions[i]
                alerts[i].append(alert)
            if person_count > 2:
                alert = "People to the " + directions[i]
                alerts[i].append(alert)
            if vehicle_count > 0:
                alert = "Vehicles to the " + directions[i]
                alerts[i].append(alert)
            if train_count > 0:
                alert = "Trains to the " + directions[i]
                alerts[i].append(alert)
           
        alert_system.alerts = alerts
        return

    def play():
        from win32com.client import Dispatch
        from pythoncom import CoInitialize

        CoInitialize()
        speak = Dispatch('SAPI.SpVoice')

        while(True):
            alerts = alert_system.alerts
            for i in alerts:
                if len(i) == 0:
                        continue
                print(i)
                for alert in i:
                    speak.Speak(alert)
            sleep(1)

    def start_play_thread():

        audio_thread.start()
        return

audio_thread = Thread(target=alert_system.play, args=() , name='AudioFeedbackThread')
audio_thread.daemon = True

if __name__ == "__main__":
    import numpy as np
    cls = [1,6,6,6,6,6,6,2,2,2,7]
    cls = np.array(cls)
    alert_system.start_play_thread()
    alert_system.check(cls)