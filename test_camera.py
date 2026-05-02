import cv2
import os
import json

print("[TEST] Checking camera...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print("[TEST] Camera opened successfully")
print("[TEST] Capturing frame...")

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Cannot capture frame")
    exit(1)

print("[TEST] Frame captured:", frame.shape)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"[TEST] Found {len(faces)} face(s)")

if len(faces) > 0:
    print("[SUCCESS] Face detected! Ready to register.")
else:
    print("[WARNING] No face detected. Make sure you're in frame.")