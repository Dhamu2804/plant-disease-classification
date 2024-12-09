from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load the model from the new location
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Validation set (if required for later use)
valid_dir = r'C:\Users\kondu\.cache\kagglehub\datasets\vipoooool\new-plant-diseases-dataset\versions\2\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
)

# Class names (for disease categories)
class_names = validation_set.class_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Convert the file to a BytesIO object
    img_bytes = file.read()
    img = image.load_img(BytesIO(img_bytes), target_size=(128, 128))  # Load the image from BytesIO
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict disease
    predictions = cnn.predict(img_array)
    result_index = np.argmax(predictions)
    model_prediction = class_names[result_index]

    # Convert image to base64 for display in HTML
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return render_template('prediction.html', prediction=model_prediction, img_str=img_str)

if __name__ == '__main__':
    app.run(debug=True)
