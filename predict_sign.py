import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
LABELS = ["Stop", "Speed Limit 50", "Yield", "No Entry", ..., "Roundabout"]  # Replace with actual label names

# Load the trained model
model = load_model("best_model.h5")

def predict_image(img_path):
    """Takes an image path and returns a prediction (class, label, probability)."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    probability = float(np.max(predictions))

    return class_index, LABELS[class_index], probability

def upload_and_predict():
    """Opens file dialog to upload an image and display prediction."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and show image
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Make prediction
    class_id, class_label, prob = predict_image(file_path)
    result_label.config(text=f"Prediction: {class_label} ({class_id})\nConfidence: {prob:.3f}")

# GUI Setup
root = tk.Tk()
root.title("Traffic Sign Predictor")
root.geometry("400x400")

btn = tk.Button(root, text="Upload Image", command=upload_and_predict)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12))
result_label.pack()

root.mainloop()
