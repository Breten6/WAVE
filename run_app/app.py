import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
import cv2
import threading
import paramiko
import dlib
def crop_to_center(image, width=640, height=480):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    start_x = max(center_x - width // 2, 0)
    start_y = max(center_y - height // 2, 0)

    end_x = min(start_x + width, image.shape[1])
    end_y = min(start_y + height, image.shape[0])

    return image[start_y:end_y, start_x:end_x]
def detect_full_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        print("Face(s) detected.")
        return True
    else:
        print("No full face detected.")
        return False

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector() 
    def initUI(self):
        self.setWindowTitle('Camera Capture and SSH Transfer')

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Image label
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Capture Button
        self.btn_capture = QPushButton('Capture Photo', self)
        self.btn_capture.clicked.connect(self.capture_image)
        self.layout.addWidget(self.btn_capture)

        # Send Button
        self.btn_send = QPushButton('Send via SSH', self)
        self.btn_send.clicked.connect(self.send_via_ssh)
        self.layout.addWidget(self.btn_send)
        self.btn_send.setEnabled(False)
    def capture_image(self):
        # Implement camera capture here
        ret, frame = self.cap.read()
        if ret:
            print(type(frame))
            new_img = crop_to_center(frame)
            cv2.imwrite('default.jpg', new_img)  
            self.show_image('default.jpg')
            booo = detect_full_face('default.jpg')
            if booo:
                self.btn_send.setEnabled(True)
            else:
                self.btn_send.setEnabled(False) 

    def show_image(self, path):
        pixmap = QPixmap(path)
        self.image_label.setPixmap(pixmap)

    def send_via_ssh(self):
        server = ('raspberrypi.local', 22)  
        t = paramiko.Transport(server)  
        t.connect(username='pi',password='raspberry')  
        sftp = paramiko.SFTPClient.from_transport(t)  

        local_file = 'default.jpg' 
        remote_file = '/home/pi/VisionPi/default.jpg'  
        sftp.put(local_file, remote_file) 

    def closeEvent(self, event):
        self.cap.release()

def main():
    app = QApplication(sys.argv)
    ex = CameraApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
