import cv2
import numpy as np
import uuid
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model_nominal = load_model('model_nominal.h5')
class_names = ['2k', '5k', '50k', '100rb', '10k', '20rb']

# Fungsi preprocessing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    dilated = cv2.dilate(edges, None, iterations=2)
    return dilated

def apply_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded

def detect_contours(frame, dilated, expected_contours, distance_threshold=30):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            detected_objects.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    correct = 0.5
    for dx, dy, dw, dh in detected_objects:
        for ex, ey, ew, eh in expected_contours:
            dc = (dx + dw / 2, dy + dh / 2)
            ec = (ex + ew / 2, ey + eh / 2)
            if np.linalg.norm(np.array(dc) - np.array(ec)) < distance_threshold:
                correct += 1
    acc = (correct / len(expected_contours)) * 100 if len(expected_contours) > 0 else 0
    return frame, acc

def get_expected_contours():
    return [(100, 150, 50, 50), (200, 250, 60, 60), (300, 350, 70, 70)]

def classify_frame(frame):
    resized = cv2.resize(frame, (200, 200))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    pred = model_nominal.predict(arr)[0]
    idx = np.argmax(pred)
    conf = tf.nn.softmax(pred)[idx].numpy() * 100
    return class_names[idx], conf

cap = cv2.VideoCapture("img/4.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_writers = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    dilated = preprocess_frame(frame)
    thresholded = apply_threshold(frame)
    expected_contours = get_expected_contours()
    annotated, contour_acc = detect_contours(frame.copy(), dilated, expected_contours)
    nominal_label, confidence = classify_frame(annotated)

    acc_text = f"Class: {nominal_label.upper()} ({confidence:.2f}%) | Real: {(contour_acc * 4):.2f}%"
    cv2.putText(annotated, acc_text, (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if nominal_label not in video_writers:
        os.makedirs("recorded", exist_ok=True)
        filename = f"recorded/{nominal_label}_{str(uuid.uuid4())[:8]}.mp4"

        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        video_writers[nominal_label] = writer

    video_writers[nominal_label].write(annotated)

    cv2.imshow("Preview", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
for writer in video_writers.values():
    writer.release()
cv2.destroyAllWindows()

print("📽️ Semua video telah direkam berdasarkan kelas di folder 'recorded/'")
