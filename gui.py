import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('model.h5')

# Define class names
class_names = ['Al Pacino', 'Angelina Jolie']

# Function to predict the class of the selected image
def predict_class():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    image = cv2.resize(image, (100, 100))  # Resize image as per your model input size
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize image data

    # Predict the class of the image
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    result_label.config(text=f"Predicted class: {predicted_class_name}")

# GUI
root = tk.Tk()
root.title("CNN Image Classifier")

# Set window size
root.geometry("400x200")

# Create a frame for image display
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Label for displaying uploaded image
label_image = tk.Label(image_frame)
label_image.pack()

# Button for uploading image
upload_button = tk.Button(root, text="Select Image", command=predict_class)
upload_button.pack()

# Label for displaying prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack()

root.mainloop()
