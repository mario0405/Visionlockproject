import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import pygame
import threading
from time import sleep
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime

# Initialize pygame mixer for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')
alarm_playing = False
alarm_thread = None
pending_unknown_detection = False
unknown_detection_start_time = 0
POPUP_DELAY = 3  # 3 seconds delay before showing popup

def play_alarm():
    global alarm_playing
    while alarm_playing:
        alarm_sound.play()
        sleep(2)

def train_model():
    print("[INFO] Training model...")
    dataset_path = "dataset"
    encodings = []
    names = []
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encodings.append(face_encodings[0])
                    names.append(person_name)
    
    print("[INFO] Serializing encodings...")
    data = {"encodings": encodings, "names": names}
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
    
    return encodings, names

def save_face_to_dataset(frame, name):
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(person_path, f"{name}_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    
    # Train model immediately after saving
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = train_model()

class FacePopup:
    def __init__(self):
        self.root = None
        self.name_var = None
        self.result = None

    def show(self):
        self.root = tk.Tk()
        self.root.title("Unknown Face Detected")
        
        window_width = 300
        window_height = 150
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        tk.Label(self.root, text="Add this face to dataset?").pack(pady=10)
        
        self.name_var = tk.StringVar()
        tk.Label(self.root, text="Enter name:").pack()
        tk.Entry(self.root, textvariable=self.name_var).pack(pady=5)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Yes", command=self.on_yes).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="No", command=self.on_no).pack(side=tk.LEFT)
        
        self.result = None
        self.root.mainloop()
        return self.result, self.name_var.get()

    def on_yes(self):
        if not self.name_var.get().strip():
            messagebox.showerror("Error", "Please enter a name")
            return
        self.result = True
        self.root.destroy()

    def on_no(self):
        self.result = False
        self.root.destroy()

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# Initialize GPIO
output = LED(14)

# Initialize variables
cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
last_unknown_time = 0
UNKNOWN_FACE_COOLDOWN = 10  # Cooldown between detections

# List of names that will trigger the GPIO pin
authorized_names = ["john", "alice", "bob"]

def process_frame(frame):
    global face_locations, face_encodings, face_names, alarm_playing, alarm_thread
    global last_unknown_time, pending_unknown_detection, unknown_detection_start_time
    
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    unknown_face_detected = False
    current_time = time.time()
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name in authorized_names:
                    authorized_face_detected = True
            else:
                unknown_face_detected = True
                if not pending_unknown_detection and current_time - last_unknown_time > UNKNOWN_FACE_COOLDOWN:
                    pending_unknown_detection = True
                    unknown_detection_start_time = current_time
                
                if pending_unknown_detection and current_time - unknown_detection_start_time >= POPUP_DELAY:
                    last_unknown_time = current_time
                    pending_unknown_detection = False
                    popup = FacePopup()
                    add_face, person_name = popup.show()
                    if add_face and person_name:
                        save_face_to_dataset(frame, person_name)
        else:
            unknown_face_detected = True
            
        face_names.append(name)
    
    if authorized_face_detected:
        output.on()
        alarm_playing = False
        pending_unknown_detection = False
    elif unknown_face_detected:
        output.off()
        if not alarm_playing:
            alarm_playing = True
            alarm_thread = threading.Thread(target=play_alarm)
            alarm_thread.daemon = True
            alarm_thread.start()
    else:
        output.off()
        alarm_playing = False
        pending_unknown_detection = False
    
    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        color = (244, 42, 3) if name == "Unknown" else (0, 255, 0)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
        elif name == "Unknown":
            cv2.putText(frame, "UNAUTHORIZED!", (left + 6, bottom + 23), font, 0.6, (0, 0, 255), 1)
            if pending_unknown_detection:
                remaining_time = POPUP_DELAY - (time.time() - unknown_detection_start_time)
                if remaining_time > 0:
                    cv2.putText(frame, f"Capturing in: {remaining_time:.1f}s", 
                              (left + 6, bottom + 46), font, 0.6, (0, 0, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

try:
    while True:
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)
        display_frame = draw_results(processed_frame)
        current_fps = calculate_fps()
        
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                    (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Video', display_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    alarm_playing = False
    if alarm_thread is not None:
        alarm_thread.join(timeout=1)
    cv2.destroyAllWindows()
    picam2.stop()
    output.off()
    pygame.mixer.quit()