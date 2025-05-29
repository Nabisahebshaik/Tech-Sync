import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import face_recognition

# === CONFIG ===
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.5         # Lower = stricter (0.4-0.6 typical)
NUM_JITTERS = 1         # To avoid dlib compatibility bug

# === Initialize attendance ===
if os.path.exists(ATTENDANCE_FILE):
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

# === Load known faces ===
known_encodings = []
known_names = []

print("🔄 Loading known faces...")
for fname in os.listdir(KNOWN_FACES_DIR):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    path = os.path.join(KNOWN_FACES_DIR, fname)
    img = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(img, num_jitters=NUM_JITTERS)
    if encs:
        known_encodings.append(encs[0])
        known_names.append(os.path.splitext(fname)[0])
        print(f"  ✔ Loaded: {fname}")
    else:
        print(f"  ⚠️  Skipped {fname}: No faces found")

if not known_encodings:
    print("❌ No valid training images found")
    exit()

print(f"✅ Loaded {len(known_encodings)} known faces")

# === Webcam setup ===
cap = cv2.VideoCapture(0)
print("🎥 Webcam active. Press Q to quit")

marked = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # Ensure memory layout
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    face_encodings = face_recognition.face_encodings(
        rgb_frame, 
        known_face_locations=face_locations,
        num_jitters=NUM_JITTERS
    )

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        color = (0, 0, 255)  # Red
        confidence = 0.0

        if True in matches:
            best_match_idx = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_idx]
            if confidence > (1 - TOLERANCE):
                name = known_names[best_match_idx]
                color = (0, 255, 0)  # Green

                # Mark attendance only once per session
                if name not in marked:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_df = pd.concat([attendance_df, pd.DataFrame([[name, timestamp]],
                                              columns=["Name", "Time"])], ignore_index=True)
                    marked.add(name)
                    print(f"✅ Marked: {name} at {timestamp}")

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw filled rectangle for text background inside the box
        label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # Draw filled rectangle at the bottom of the face box
        cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width, bottom), color, cv2.FILLED)
        # Put text inside the filled rectangle
        cv2.putText(frame, label, (left, bottom - 5), font, font_scale, (255, 255, 255), thickness)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
attendance_df.to_csv(ATTENDANCE_FILE, index=False)
cap.release()
cv2.destroyAllWindows()
print(f"📊 Attendance data saved to {ATTENDANCE_FILE}")
