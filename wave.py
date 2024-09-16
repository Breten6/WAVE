import time
import cv2
import sys
import os
import queue
import PWMServo
from wave_function import *
import math
import numpy as np
import mediapipe as mp
import pygame
import requests
import json
mode = 0
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
                    max_num_hands =1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

send_email = ''
recv_email = ''
passwd = ''

def get_email_data():
    global send_email, passwd, recv_email

    send_email, passwd, recv_email = read_data()

cap = ''
import math
pygame.mixer.init()
pygame.mixer.music.load("Triggers.mp3")


def camera_open():
    global cap
    
    try:
        if cap != '':
            try:
                cap.release()
            except Exception as e:
                print(e)
        cap = cv2.VideoCapture(-1)
        cap.set(12, 45)
        time.sleep(0.001)
    except BaseException as e:
        print('open camera error:',e)

def camera_close():
    global cap
    
    try:
        time.sleep(0.1)
        cap.release()
        time.sleep(0.01)
    except BaseException as e:
        print('close camera error:', e)
        
frame_copy = image = None

def wave_func():
    global cap
    global  image, frame_copy
    while True:
        if frame_copy is not None: 

            image = WAVE(frame_copy)
            frame_copy = None
            time.sleep(0.001)
        else:
            time.sleep(0.01)

def camera_task():
    global image, frame_copy,  cap
    while True:
        

        try:
            ret, orgframe = cap.read()
            if ret:
                frame_flip = cv2.flip(orgframe, 1)
                frame_copy = frame_flip
                img_tmp = image if image is not None else frame_flip
                cv2.imshow('image',img_tmp)
                cv2.waitKey(1)                    
                time.sleep(0.001)
            else:
                cap = cv2.VideoCapture(-1)
                time.sleep(0.01)

        except BaseException as e:
            print('error on camera', e)


threading.Thread(target=wave_func, daemon=True).start()
threading.Thread(target=camera_task, daemon=True).start()

if __name__ == '__main__':
    send_email = ''
    recv_email = ''
    passwd = ''
    camera_open()
    tmp_mode = 0 
    
    interrupt = False
    def signal_handle(signal, frame): 
        global interrupt
        
        interrupt = True
        print('end')

    while True:
        if interrupt:
            break
        else:
            time.sleep(1)
