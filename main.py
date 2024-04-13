import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from keras.callbacks import ReduceLROnPlateau

sns.set()

image = tf.keras.preprocessing.image
ImageDataGenerator = image.ImageDataGenerator

# Getting names of the classes we have
data_directory = pathlib.Path('data/raw/Fruits Classification')
train_directory = data_directory / 'train'
test_directory = data_directory / 'test'
class_names = sorted([item.name for item in train_directory.glob('*')][:5])

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

train_data = data_gen.flow_from_directory(train_directory,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          subset='training',
                                          class_mode='binary')
val_data = data_gen.flow_from_directory(train_directory,
                                        target_size=(224, 224),
                                        batch_size=32,
                                        subset='validation',
                                        class_mode='binary')

# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.00001)
]

# Train the model
classifier = model.fit(
    train_data, batch_size=32, epochs=5, validation_data=val_data, callbacks=callbacks
)

# Evaluate the model
model.evaluate(val_data)

# Plot model performance graphs
def model_performance_graphs():
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].plot(classifier.epoch, classifier.history['accuracy'], label='acc')
    axes[0].plot(classifier.epoch, classifier.history['val_accuracy'], label='val_acc')
    axes[0].set_title('Accuracy vs Epochs', fontsize=20)
    axes[0].set_xlabel('Epochs', fontsize=15)
    axes[0].set_ylabel('Accuracy', fontsize=15)
    axes[0].legend()

    axes[1].plot(classifier.epoch, classifier.history['loss'], label='loss')
    axes[1].plot(classifier.epoch, classifier.history['val_loss'], label="val_loss")
    axes[1].set_title("Loss Curve", fontsize=18)
    axes[1].set_xlabel("Epochs", fontsize=15)
    axes[1].set_ylabel("Loss", fontsize=15)
    axes[1].legend()

    plt.show()

model_performance_graphs()

# Save the model
modelpath = "out/models/model.h5"
model.save(modelpath)

# Load the model
loaded_model: tf.keras.Model = tf.keras.models.load_model(modelpath)

# Preprocess images
def preprocess_images(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_array / 255.0
    return img_preprocessed

# Load test images
test_images = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
    test_directory, target_size=(224, 224), shuffle=False, batch_size=100
)

X_test, y_test = test_images.next()
y_test = np.argmax(y_test, axis=1)

# Predictions
y_pred = np.argmax(loaded_model.predict(X_test), axis=1)

# Confusion matrix
cmat = {i: {j: 0 for j in range(5)} for i in range(5)}
for t, p in zip(y_test, y_pred):
    cmat[t][p] += 1
cmat = pd.DataFrame(cmat).T

plt.figure(figsize=(10, 10))
sns.heatmap(cmat, annot=True, cmap="Blues")
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.show()
