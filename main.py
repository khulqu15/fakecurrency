import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    dilated = cv2.dilate(edges, None, iterations=2)
    return image, dilated

def apply_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresholded_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded_image

def detect_contours(image, dilated_edges, expected_contours, distance_threshold=30):
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
    return image, accuracy

def get_expected_contours():
    return [
        (100, 150, 50, 50),
        (200, 250, 60, 60),
        (300, 350, 70, 70)
    ]

def detect_security_features(image_path):
    expected_contours = get_expected_contours()

    image, dilated_edges = preprocess_image(image_path)
    detected_image, accuracy = detect_contours(image.copy(), dilated_edges, expected_contours)

    thresholded_image = apply_threshold(image)

    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    img_resized = cv2.resize(detected_image, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    model_accuracy = prediction[0][0] * 100

    accuracy_text = f"Model Accuracy: {model_accuracy:.2f}% | Contour Accuracy: {accuracy:.2f}%"
    cv2.putText(detected_image, accuracy_text, (10, detected_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2, cv2.LINE_AA)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title("Dilated Edges")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(dilated_edges, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box and Accuracy")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

image_path = "img/2.jpg" 
detect_security_features(image_path)
