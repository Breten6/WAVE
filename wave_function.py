#!/usr/bin/python3
#coding=utf8
import sys
import gc
if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
import csv
import os
import cv2
import face_recognition
import numpy as np
import time
import math
import datetime
import threading
import smtplib
import re
import dlib
import pid
import PWMServo
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart 
from email.mime.image import MIMEImage
import cv2
import mediapipe as mp
import pygame
import requests
import json
recognize_flag = 0
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
                    max_num_hands =1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
def encodeFace(image, face_landmarks):
  
  face_encoder = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

  face = dlib.get_face_chip(image, face_landmarks)
   
  encodings = np.array(face_encoder.compute_face_descriptor(face))
  return encodings
def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


servo1_face_track = 1700
servo2_face_track = 1500

capture_lock = False
dis_ok_face = False
action_finish_face = True
def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  

    if len(point_a) == len(point_b) == len(point_c) == 3:

        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2] 
    else:
        a_z, b_z, c_z = 0,0,0

    x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
    x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)


    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) *(math.sqrt(x2**2 + y2**2 + z2**2))) 
    B = math.degrees(math.acos(cos_b)) 
    return B
# Action
def track():
    global servo1_face_track, servo2_face_track
    global dis_ok_face, action_finish_face
    global capture_lock
    while True:
        dis_ok_face = False
        action_finish_face = False
        # capture_lock = True
        PWMServo.setServo(1, servo1_face_track, 20)
        PWMServo.setServo(2, servo2_face_track, 20)
        # capture_lock = False
        time.sleep(0.01)
        action_finish_face = True
        time.sleep(0.1)


 #Thread start
sv_track = threading.Thread(target=track)
sv_track.setDaemon(True)
sv_track.start()  


send_count = 0
msg  = None
mss_ =None
smtpObject= None    
smtp_server = "smtp.qq.com"
smtp_port = 465
def SendEmail(Frame, sender, warrant, receiver, number):
    global send_count
    global status
    global mss_
    global msg

    if send_count == 0:
        msg = MIMEMultipart('related')
        msg["From"] = Header("WAVE", "utf-8")
        msg["To"]   = Header(receiver, "utf-8")
        msg["Subject"] = Header("Face Detect!", "utf-8")
        mss_ = ""
        
        mss =''''''
        mss = re.split(r'\n',mss)[1:]
        for i in mss:
            mss_ += "<p>" + i + "</p>"
        
    send_count += 1
    cv2.imwrite(str(send_count) + '.jpg', Frame)
    time.sleep(0.05)
    img = open(str(send_count) + '.jpg',"rb")
    img_data = img.read()
    img.close()
    img = MIMEImage(img_data)
    img.add_header('Content-ID', str(send_count))
    msg.attach(img)
    
    mss_ += "<img src='cid:" + str(send_count) + "'/><p></p>"
    if send_count >= number:
        send_count = 0
#         print(mss_)
        message = MIMEText(mss_,"html","utf-8")
        msg.attach(message)
        try:
            #print(sender, warrant, receiver)
            smtpObject = smtplib.SMTP_SSL(smtp_server, smtp_port)
            smtpObject.login(sender, warrant)
            smtpObject.sendmail(sender , [receiver] , msg.as_string())
            print ("send success!")
            smtpObject.quit()
            return 'send_ok'
        except smtplib.SMTPException as e:
            error_code = str(e.smtp_code)
            error_msg = str(e.smtp_error).split('.')[0][2:]
            print ("send fail! ",error_code, error_msg)
            smtpObject.quit()
            return 'send fail!\nerror_code:{}\nerror_msg:{}'.format(error_code, error_msg)

DNN = "TF"
if DNN == "CAFFE":
    modelFile = "/home/pi/wave/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "/home/pi/wave/models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "/home/pi/wave/models/opencv_face_detector_uint8.pb"
    configFile = "/home/pi/wave/models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

count = 0
limit_up = 100
limit_down = 1
light = 1
light_on = {"on":True,"bri":1}
light_off = {"on":False}
debug = False
light_url = "http://192.168.0.198/api/p1csEbKFRYbAqmqSNNDs93MaLvZv6OXi6DhXBjW8/lights/1/state"
joint_list = [[4,3, 2],[7, 6,5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
pygame.mixer.init()
pygame.mixer.music.load("Triggers.mp3")
volume = 1.0
open_hand = 0
close_hand = 0
point_up = 0
point_down = 0
thumb_up = 0
thumb_down = 0
play_flag = 0
pygame.mixer.music.set_volume(volume)
def hand_track(frame):
    global hands, light, volume,play_flag, capture_lock, open_hand, close_hand, point_down, point_up, thumb_down, thumb_up

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
    finger = [2,2,2,2,2]
    angles = [0,0,0,0,0]
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
        innn = 0
        for joint in joint_list:
            a = np.array([hand_landmarks.landmark[joint[0]].x, hand_landmarks.landmark[joint[0]].y])
            b = np.array([hand_landmarks.landmark[joint[1]].x, hand_landmarks.landmark[joint[1]].y])
            c = np.array([hand_landmarks.landmark[joint[2]].x, hand_landmarks.landmark[joint[2]].y])

            angle = cal_angle(a,b,c)
            limit_angle = 165
            if innn ==0:
                limit_angle = 148
            if angle > 180.0:
                angle = 360 - angle
            angles[innn] = angle
            if angle < limit_angle:
                finger[innn] = 0
            if a[1] < b[1] < c[1]:
                if angle > limit_angle:
                    finger[innn] = 1
            elif a[1] > b[1] > c[1]:
                if angle > limit_angle:
                    finger[innn] = -1
            innn+=1
    print(finger)
    print(angles)
    if all(f == 0 for f in finger):
        print("close hand")
        open_hand = 0
        point_up = 0
        point_down = 0
        thumb_up = 0
        thumb_down = 0
        close_hand += 1 
        if close_hand >2:
            pygame.mixer.music.pause()
            close_hand = 0
    elif all(f == 1 for f in finger):
        print("open hand")
        close_hand = 0
        point_up = 0
        point_down = 0
        thumb_up = 0
        thumb_down = 0
        open_hand += 1
        if open_hand >3:
            if play_flag == 0:                        
                pygame.mixer.music.play()
                play_flag = 1
            else:
                pygame.mixer.music.unpause()
            open_hand=0
        elif open_hand > 15:
            open_hand=0
            pass
    elif finger[1] == 1 and all(f == 0 for f in finger[2:]):
        open_hand = 0
        close_hand = 0
        point_down = 0
        thumb_up = 0
        thumb_down = 0
        point_up +=1
        print("point up")
        if point_up > 2:
            volume = volume + 0.1 if volume + 0.1 < 1.0 else 1.0
            pygame.mixer.music.set_volume(volume)
    elif finger[1] == -1 and all(f == 0 for f in finger[2:]):
        open_hand = 0
        point_up = 0
        close_hand = 0
        thumb_up = 0
        thumb_down = 0
        print("point down")
        point_down += 1
        if point_down > 2:
            
            volume = volume - 0.1 if volume - 0.1 > 0.0 else 0.0
            pygame.mixer.music.set_volume(volume)
    elif finger[0] == 1 and all(f != 1 for f in finger[1:]):
        print("thumb up")
        open_hand = 0
        point_up = 0
        point_down = 0
        close_hand = 0
        thumb_down = 0
        thumb_up += 1
        if thumb_up > 3:
            try:
                light = light+10 if light +10 < limit_up else limit_up
                signal = {"on":True,"bri":light}
                r = requests.put(light_url, json.dumps(signal), timeout=5)
                r.raise_for_status() 
                print("Request successful")
            except requests.exceptions.RequestException as e:
                pass
    elif finger[0] == -1 and all(f == 0 for f in finger[1:]):
        print("thumb down")
        open_hand = 0
        point_up = 0
        point_down = 0
        thumb_up = 0
        close_hand = 0
        thumb_down+=1
        if thumb_down > 3:
            try:
                light = light-10 
                signal = {"on":True,"bri":1}
                if light < limit_down:
                    light = limit_down
                    signal = {"on":False}
                else:
                    signal = {"on":True,"bri":light}
                r = requests.put(light_url, json.dumps(signal), timeout=5)
                r.raise_for_status()  
                print("Request successful")
            except requests.exceptions.RequestException as e:
                pass
    else:
        open_hand = 0
        point_up = 0
        point_down = 0
        thumb_up = 0
        close_hand = 0
        thumb_down = 0
    return frame

detector = dlib.get_frontal_face_detector() #get face detector
predictor_path = ("/home/pi/wave/shape_predictor_5_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

servo1_pid3 = pid.PID(P=0.6, I=0.5, D=0.01)
servo2_pid4 = pid.PID(P=0.8, I=0.5, D=0.01)


reference_image = cv2.imread("default.jpg")
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_faces = detector(reference_gray)
reference_shape = predictor(reference_gray, reference_faces[0]) 
def WAVE(frame):
    global servo1_face_track, servo2_face_track, recognize_flag
    global dis_ok_face, action_finish_face
    global net
    global reference_shape
    image = frame
    tmp_img = frame
    frameOpencvDnn = cv2.resize(frame,(160 , 120), interpolation = cv2.INTER_CUBIC)

    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1, (150, 150), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    max_area = 0 #The box area of max face
    max_face = (0,0,0,0)
    cropped = (0,0,0,0)
    cropped_index = list(cropped)
    for i in range(detections.shape[2]):
        reason = 0
        confidence = detections[0, 0, i, 2]
        if confidence > 0.35: #face recognition threshold 
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            area_tmp = ((y2 - y1 + 1) * (x2 - x1 + 1))
            if max_area < area_tmp: #Compare to judge whether it is the max face.
                max_area = area_tmp
                max_face = (x1, y1, x2, y2)
            rx1 = int(leMap(x1, 0,160, 0, 640)) # Convert coordinates to 640 * 480
            ry1 = int(leMap(y1, 0,120, 0, 480))
            rx2 = int(leMap(x2, 0,160, 0, 640))
            ry2 = int(leMap(y2, 0,120, 0, 480))

            cropped=(rx1, ry1, rx2, ry2)
            cropped_index = list(cropped)
            cw = cropped_index[2] - cropped_index[0]
            ch = cropped_index[3] - cropped_index[1]
            cropped_index[0] = int(cropped_index[0] - 0.6*cw) if cropped_index[0]-0.6*cw > 0 else 0
            cropped_index[1] = int(cropped_index[1] - 0.4*ch) if cropped_index[1]-0.4*ch > 0 else 0
            cropped_index[2] = int(cropped_index[2] + 0.6*cw) if cropped_index[2]+0.6*cw < 640 else 640
            cropped_index[3] = int(cropped_index[3] + 0.4*ch) if cropped_index[1]+0.4*ch < 480 else 480
            cropped_face = frame[cropped_index[1]:cropped_index[3], cropped_index[0]:cropped_index[2]]
            detect_image = cv2.resize(cropped_face, (160, 120), interpolation=cv2.INTER_CUBIC)
            detect_image_gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
            detect_image_faces = detector(detect_image_gray)
            if len(detect_image_faces) > 0:
                detected_shape = predictor(detect_image_gray, detect_image_faces[0])
                face1 = encodeFace(reference_image, reference_shape)
                face2 = encodeFace(detect_image, detected_shape)
                similarity = np.linalg.norm(face1-face2)
                print("similari is ",similarity)
                if similarity < 0.5:
                    print("matched!",i)
                    recognize_flag = recognize_flag+1 if recognize_flag +1 <6 else 5
                    image = cv2.rectangle(frame, (cropped_index[0], cropped_index[1]), (cropped_index[2], cropped_index[3]), (0, 255, 0), 2) #Frame the face on the image
                    max_area = area_tmp
                    max_face = (x1, y1, x2, y2)
                    break
                    # image = cv2.rectangle(image, (340, 20), (620, 460), (0, 255, 0), 2)
                else:
                    reason = 1
                    print("not match",i)

            else:
                print("No faces detected in the cropped image.")
            if reason:
                recognize_flag = recognize_flag-1 if recognize_flag -1 >0 else 0
            cv2.rectangle(image, (cropped_index[0], cropped_index[1]), (cropped_index[2], cropped_index[3]), (0, 255, 0), 2) #Frame the face on the image 
    img_center_y = 60 #the center y of image 
    img_center_x = 80
    if max_area != 0: #control servo to track
        center_x, center_y = (max_face[0] + int((max_face[2] - max_face[0]) / 2), max_face[1] + int((max_face[3] - max_face[1]) /2)) #The coordinate of the max face
        #
        #up and down refresh pid 
 
        if abs(img_center_y - center_y) < 20:
            pass
        else:
            servo1_pid3.SetPoint = img_center_y #Stop pid operation within a certain range from the center
            servo1_pid3.update(center_y)
            tmp = int(servo1_face_track - servo1_pid3.output)
            tmp = tmp if tmp > 500 else 500
            servo1_face_track = tmp if tmp < 1950 else 1950#舵机角度限位  servo angle limit
        #
        #left and right refresh pid
        if abs(img_center_x - center_x) < 50 :
            pass
        else:
            servo2_pid4.SetPoint = img_center_x #Stop pid operation within a certain range from the center
            servo2_pid4.update(2 * img_center_x - center_x) 
            tmp = int(servo2_face_track + servo2_pid4.output)
            tmp = tmp if tmp > 500 else 500
            servo2_face_track = tmp if tmp < 2500 else 2500 #servo angle limit
        

        # print(servo1_face_track,servo2_face_track)
        if action_finish_face:
            dis_ok_face = True
        print(action_finish_face)
    if dis_ok_face:
        if recognize_flag > 0:
            image = hand_track(image)
    # Handle the situation when no faces are detected
    gc.collect()

    return image
