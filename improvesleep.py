import cv2
import dlib
from imutils.video import VideoStream
import time
import numpy as np
import argparse
import imutils
from scipy.spatial import distance as dist
import winsound
from imutils import face_utils

def play_alarm_sound():
    winsound.Beep(500, 1000)  # Beep at 500 Hz for 1000 ms

def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-f", "--face_cascade_path", type=str, default='haarcascade_frontalface_default.xml', help="path to face cascade file")
ap.add_argument("-e", "--eye_cascade_path", type=str, default='haarcascade_eye_tree_eyeglasses.xml', help="path to eye cascade file")
ap.add_argument("-p", "--predictor_path", type=str, default='shape_predictor_68_face_landmarks.dat', help="path to dlib face landmark predictor")
args = vars(ap.parse_args())

vs = VideoStream(src=args["webcam"]).start()
time.sleep(2.0)

face_cascade = cv2.CascadeClassifier(args["face_cascade_path"])
eye_cascade = cv2.CascadeClassifier(args["eye_cascade_path"])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["predictor_path"])

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10

alarm_status = False
COUNTER = 0

while True:
    frame = vs.read()
    if frame is None:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected")
    else:
        print(f"Detected {len(faces)} face(s)")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)

        if len(eyes) == 0:
            print("Eyes closed or not detected!")
        else:
            print(f"Detected {len(eyes)} eye(s)")

        shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
        shape_np = face_utils.shape_to_np(shape)

        leftEye = shape_np[36:42]
        rightEye = shape_np[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    play_alarm_sound()
                    print("Drowsiness alert triggered!")
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER > 0:
                print(f"Eyes reopened. Alert counter reset after {COUNTER} frames.")
            COUNTER = 0
            alarm_status = False

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
