import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a traffic sign recognition model.")
parser.add_argument("data_dir", help="Path to the directory containing the dataset.")
parser.add_argument("--model", help="Path to save the trained model.", default="model.h5")
args = parser.parse_args()

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
EPOCHS = 10  # Adjust the number of epochs as needed

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """
    images = []
    labels = []
    
    # Iterate over category directories
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        # Iterate over images in category directory
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            images.append(img)
            labels.append(category)
    
    return np.array(images), np.array(labels)


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data
images, labels = load_data(args.data_dir)

# Split data into training and testing sets
TEST_SIZE = 0.4
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=TEST_SIZE
)

# Normalize image data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = get_model()

# Train the model
print("Starting training...")
model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))
print("Training complete.")

# Save the model
if args.model:
    model.save(args.model)
    print(f"Model saved to {args.model}")
