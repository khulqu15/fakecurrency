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

def detect_security_features(image_path):
    image, dilated_edges = preprocess_image(image_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dilated_edges, cmap='gray')
    plt.title("Dilated Edges")
    plt.axis('off')
    plt.show()
    
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    image_input = cv2.imread(image_path)
    img_resized = cv2.resize(image_input, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    accuracy = prediction[0][0] * 100

    accuracy_text = f"Prediction Accuracy: {accuracy:.2f}%"
    cv2.putText(image_input, accuracy_text, (10, image_input.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    plt.title("Detected Features with AI")
    plt.axis('off')
    plt.show()

image_path = "img/2.jpg"
detect_security_features(image_path)
