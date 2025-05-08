import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Config:
    img_height = 200
    img_width = 200
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3

dir_path = 'nominal'

dataset = tf.keras.utils.image_dataset_from_directory(
    dir_path, 
    seed=42,
    image_size=(Config.img_height, Config.img_width),
    batch_size=10
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path, 
    subset='training', 
    validation_split=0.2,
    seed=42,
    image_size=(Config.img_height, Config.img_width),
    batch_size=Config.batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path, 
    subset='validation',
    validation_split=0.2,
    seed=42,
    image_size=(Config.img_height, Config.img_width),
    batch_size=Config.batch_size
)

currency_nominal = dataset.class_names
num_of_classes = len(currency_nominal)

class myCallback(tf.keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.90:
            print("\nStop training, accuracy > 90%")
            self.model.stop_training = True

callback = myCallback()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(Config.img_height, Config.img_width, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_of_classes)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=Config.learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Config.epochs,
    callbacks=[callback]
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_title("Accuracy")
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.legend()
ax2.set_title("Loss")
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.legend()
plt.tight_layout()
plt.show()

model.save("model_nominal.h5")


y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds_labels = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=currency_nominal)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()