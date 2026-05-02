import os
import sys
import time
import json
import glob
import threading
from datetime import datetime

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[FACIAL] OpenCV not installed. Run: pip install opencv-python")

FACIAL_AUTH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "facial_auth.json"
)
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3
known_names = []
known_encodings = []
auth_attempts = 0
facial_auth_enabled = False


def load_known_faces():
    global known_encodings, known_names
    try:
        if os.path.exists(FACIAL_AUTH_FILE):
            with open(FACIAL_AUTH_FILE, "r") as f:
                data = json.load(f)
                known_names = data.get("names", [])
                encodings = data.get("encodings", [])
                known_encodings = [np.array(e, dtype=np.float64) for e in encodings]
                return True
    except Exception as e:
        print(f"[FACIAL] Error loading faces: {e}")
    return False


def save_known_face(name, encoding):
    try:
        data = {"names": [], "encodings": []}
        if os.path.exists(FACIAL_AUTH_FILE):
            with open(FACIAL_AUTH_FILE, "r") as f:
                data = json.load(f)
        
        data["names"] = data.get("names", []) + [name]
        data["encodings"] = data.get("encodings", []) + [encoding.tolist()]
        
        with open(FACIAL_AUTH_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[FACIAL] Error saving face: {e}")
        return False


def get_face_descriptor(frame, face_rect):
    x, y, w, h = face_rect
    face_roi = frame[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (100, 100))
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return gray.flatten()


def capture_and_encode():
    if not CV2_AVAILABLE:
        return None
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[FACIAL] Cannot access camera")
        return None
    
    time.sleep(1)
    
    ret, frame = video_capture.read()
    video_capture.release()
    
    if not ret:
        print("[FACIAL] Failed to capture frame")
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return get_face_descriptor(frame, (x, y, w, h))
    
    return None


def register_new_face(name):
    print(f"[FACIAL] Registering new face: {name}")
    print("[FACIAL] Look at the camera and press any key when ready...")
    encoding = capture_and_encode()
    
    if encoding is not None:
        if save_known_face(name, encoding):
            load_known_faces()
            print(f"[FACIAL] Face registered: {name}")
            return True
    
    print("[FACIAL] Could not detect face. Try again with better lighting.")
    print("[FACIAL] Or manually add encoding to facial_auth.json")
    return False


def register_face_auto(name):
    print(f"[FACIAL] Auto-registering: {name}")
    encoding = capture_and_encode()
    
    if encoding is not None:
        if save_known_face(name, encoding):
            load_known_faces()
            print(f"[FACIAL] Face registered: {name}")
            return True
    
    return False


def verify_face():
    if not known_encodings:
        return False, None
    
    if not CV2_AVAILABLE:
        return False, None
    
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("[FACIAL] Camera busy or not available")
            return False, None
        
        time.sleep(0.5)
        
        ret, frame = video_capture.read()
        video_capture.release()
        
        if not ret:
            return False, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, None
        
        x, y, w, h = faces[0]
        current_encoding = get_face_descriptor(frame, (x, y, w, h))
        
        best_match = None
        best_score = float('inf')
        
        for i, known_enc in enumerate(known_encodings):
            diff = np.linalg.norm(known_enc - current_encoding)
            if diff < best_score:
                best_score = diff
                best_match = i
        
        if best_score < 8000000:
            return True, known_names[best_match]
        
        return False, None
    
    except Exception as e:
        print(f"[FACIAL] Verify error: {e}")
        return False, None
    current_encoding = get_face_descriptor(frame, (x, y, w, h))
    
    best_match = None
    best_score = float('inf')
    
    for i, known_enc in enumerate(known_encodings):
        diff = np.linalg.norm(known_enc - current_encoding)
        if diff < best_score:
            best_score = diff
            best_match = i
    
    if best_score < 8000000:
        return True, known_names[best_match]
    
    return False, None


def authentication_loop(max_attempts=MAX_RETRY_ATTEMPTS, on_success=None):
    global auth_attempts, facial_auth_enabled
    
    if not CV2_AVAILABLE:
        print("[FACIAL] Module not available")
        return False
    
    if not load_known_faces() or not known_encodings:
        print("[FACIAL] No registered faces. Run 'register-face <name>' first.")
        return False
    
    print(f"[FACIAL] Starting facial authentication... (max {max_attempts} attempts)")
    auth_attempts = 0
    facial_auth_enabled = True
    
    while auth_attempts < max_attempts and facial_auth_enabled:
        auth_attempts += 1
        print(f"[FACIAL] Attempt {auth_attempts}/{max_attempts} - Looking for face...")
        
        recognized, name = verify_face()
        
        if recognized:
            print(f"[FACIAL] Welcome, {name}!")
            facial_auth_enabled = False
            
            if on_success:
                on_success(name)
            return True
        
        print(f"[FACIAL] Face not recognized. Retrying in {RETRY_DELAY_SECONDS}s...")
        time.sleep(RETRY_DELAY_SECONDS)
    
    print("[FACIAL] Authentication failed after maximum attempts")
    facial_auth_enabled = False
    return False


def authentication_thread(max_attempts=MAX_RETRY_ATTEMPTS, on_success=None):
    thread = threading.Thread(
        target=authentication_loop,
        args=(max_attempts, on_success),
        daemon=True
    )
    thread.start()
    return thread


def cancel_auth():
    global facial_auth_enabled
    facial_auth_enabled = False
    print("[FACIAL] Authentication cancelled")


def is_authenticated():
    return auth_attempts > 0 and not facial_auth_enabled


def get_auth_status():
    return {
        "enabled": facial_auth_enabled,
        "attempts": auth_attempts,
        "max_attempts": MAX_RETRY_ATTEMPTS,
        "known_faces": len(known_names),
    }


def start_preview():
    if not CV2_AVAILABLE:
        return
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    video_capture = cv2.VideoCapture(0)
    print("[FACIAL] Camera preview - Press 'q' to quit")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("JARVIS Facial Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


def facial_auth_main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Facial Recognition")
    parser.add_argument("command", choices=["verify", "register", "preview", "status"], nargs="?")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--attempts", type=int, default=MAX_RETRY_ATTEMPTS)
    
    args = parser.parse_args()
    
    if not CV2_AVAILABLE:
        print("Error: Install opencv-python")
        sys.exit(1)
    
    if args.command == "register":
        name = args.name or input("Enter name: ").strip()
        if register_new_face(name):
            print(f"Registered: {name}")
        else:
            print("Failed to register")
    
    elif args.command == "verify":
        success, name = verify_face()
        if success:
            print(f"Verified: {name}")
        else:
            print("Not recognized")
    
    elif args.command == "preview":
        start_preview()
    
    elif args.command == "status":
        if load_known_faces():
            print(f"Known faces: {known_names}")
        else:
            print("No faces registered")
    
    else:
        load_known_faces()
        if not known_names:
            print("No faces. Run: python facial_auth.py register <name>")
        else:
            print("Faces:", known_names)


if __name__ == "__main__":
    facial_auth_main()