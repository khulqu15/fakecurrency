import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    dilated = cv2.dilate(edges, None, iterations=2)
    return frame, dilated

def apply_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    _, thresholded_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded_frame

def detect_contours(frame, dilated_edges, expected_contours, distance_threshold=30):
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    correct_detections = 0.5
    for detected in detected_objects:
        dx, dy, dw, dh = detected
        for expected in expected_contours:
            ex, ey, ew, eh = expected

            detected_center = (dx + dw / 2, dy + dh / 2)
            expected_center = (ex + ew / 2, ey + eh / 2)

            distance = np.linalg.norm(np.array(detected_center) - np.array(expected_center))

            if distance < distance_threshold:
                correct_detections += 1

    accuracy = (correct_detections / len(expected_contours)) * 100 if len(expected_contours) > 0 else 0
    return frame, accuracy

def get_expected_contours():
    return [
        (100, 150, 50, 50),
        (200, 250, 60, 60),
        (300, 350, 70, 70)
    ]

def detect_security_features(frame):
    expected_contours = get_expected_contours()

    frame, dilated_edges = preprocess_image(frame)
    detected_frame, accuracy = detect_contours(frame.copy(), dilated_edges, expected_contours)

    thresholded_frame = apply_threshold(frame)

    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    img_resized = cv2.resize(detected_frame, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    model_accuracy = prediction[0][0] * 100

    accuracy_text = f"Model Accuracy: {model_accuracy:.2f}% | Contour Accuracy: {accuracy:.2f}%"
    cv2.putText(detected_frame, accuracy_text, (10, detected_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return detected_frame, thresholded_frame, dilated_edges

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 480)) 

        detected_frame, thresholded_frame, dilated_edges = detect_security_features(resized_frame)

        cv2.imshow("Original Frame", resized_frame)
        cv2.imshow("Thresholded Image", thresholded_frame)
        cv2.imshow("Dilated Edges", dilated_edges)
        cv2.imshow("Bounding Box and Accuracy", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = "img/2.mp4"
process_video(video_path)
