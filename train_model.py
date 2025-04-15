import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
MODEL_DIR = r"C:\ML_Model"
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.keras")
DATASET_PATH = r"C:/Users/sivab/Downloads/dataset"

# Create model directory if it doesn’t exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Data augmentation
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

# Load training and validation data
train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Train"), target_size=(150, 150), batch_size=32, class_mode="categorical", subset="training"
)
val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Train"), target_size=(150, 150), batch_size=32, class_mode="categorical", subset="validation"
)

# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
EPOCHS = 10  # Adjust based on dataset size
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")
