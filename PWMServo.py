import pigpio
import time
from HwServo import PWM_Servo

Servos = ()
pi = None

def setServo(servoId, pos, time):
    if servoId < 1 or servoId > 2:
        return
    if pos > 2500:
        pos = 2500
    elif pos < 500:
        pos = 500
    if time > 30000:
        time = 30000
    elif time < 20:
        time = 20
    else:
        pass
    Servos[servoId - 1].setPosition(pos, time)

def setDeviation(servoId, d):
    if servoId < 1 or servoId > 2:
        return
    if d < -300 or d > 300:
        return
    Servos[servoId - 1].setDeviation(d)

def initLeArm(d):
    global Servos
    global pi
    pi = pigpio.pi()
    servo1 = PWM_Servo(pi, 5, deviation=d[0], control_speed=True)  # slot 7
    servo2 = PWM_Servo(pi, 13, deviation=d[1], control_speed=True)   # 6
    Servos = (servo1, servo2)

d = [0, 0]
# init 
initLeArm(d)

if __name__ == '__main__':
    setServo(6, 1500, 500)
    setServo(7, 1500, 500)
    time.sleep(0.5)
    setServo(6, 1000, 500)
    setServo(7, 1000, 500)
    time.sleep(0.5)
