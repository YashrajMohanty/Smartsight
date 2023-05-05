from time import sleep
from threading import Thread
from numpy import asarray


class alert_system:

    alerts = []
    
    def check(cls, distances, obs_flag=False):

        if len(cls) == 0:
            return []

        #default filtered classes: [0,1,2,3,5,7,13,15,16,17,18,19,57] nc: 13
        #custom model classes: ['bicycle', 'bus', 'car', 'cat', 'dog', 'motorcycle', 'person', 'train', 'truck'] nc: 9
        cls = cls.astype(int)
        cls = cls.tolist()
        for i in range(len(cls)):
            if cls[i] == 0:
                cls[i] = 'person'
                continue
            for j in [1,2,3,5,7]:
                if cls[i] == j:
                    cls[i] = 'vehicle'
                    continue
            for j in [13,57]:
                if cls[i] == j:
                    cls[i] = 'bench'
                    continue
            for j in [15,16,17,18,19]:
                if cls[i] == j:
                    cls[i] = 'animal'
                    continue
        
        obj_list = []

        animal_count = 0
        person_count = 0
        vehicle_count = 0
        bench_count = 0

        alert_list = []

        for i in range(len(distances)):
            if distances[i] > 8: # only objects closer than this distance(meters) will be alerted of
                continue
            else:
                obj_list.append(str(cls[i]))

            if len(obj_list) == 0:
                continue

            for i in obj_list:
                if i == 'animal':
                    animal_count += 1
                if i == 'person':
                    person_count += 1
                if i == 'bench':
                    bench_count += 1
                if i == 'vehicle':
                    vehicle_count += 1

        if obs_flag:
            alert = "Obstruction"
            alert_list.append(alert)

        if animal_count == 1:
            alert = "Animal"
            alert_list.append(alert)
        elif animal_count > 1:
            alert = "Animals"
            alert_list.append(alert)  

        if person_count == 1:
            alert = "Person"
            alert_list.append(alert)
        elif person_count > 1:
            alert = "People"
            alert_list.append(alert)

        if vehicle_count > 0:
            alert = "Vehicle"
            alert_list.append(alert)

        if bench_count > 0:
            alert = "Bench"
            alert_list.append(alert)

        alert_system.alerts = alert_list
        return
    

    def play():
        from win32com.client import Dispatch
        from pythoncom import CoInitialize

        CoInitialize()
        speak = Dispatch('SAPI.SpVoice')
        speak.Rate = 1.8

        while(True):
            alerts = alert_system.alerts
            if len(alerts) == 0:
                sleep(1)
                continue
            print(alerts)
            alert = "".join(alerts)
            speak.Speak(alert)
            #os.system("espeak -s 155 -a 200 " + alert + "") # for espeak on ubuntu
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