# some utilities
import os
import numpy as np
from util import base64_to_pil,upload_image
import base64
from PIL import Image,ImageOps
from io import BytesIO
# Flask
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from flask import Flask, request, jsonify
import io
import tensorflow as tf
from PIL import Image

# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
Model_json = ".json"
Model_weigths = ".h5"


# Declare a flask app
app = Flask(__name__)




@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
      # Get the image data from the POST request
        image_data = request.files['image'].read()

        # Convert the image data to a PIL Image object
        image = Image.open(io.BytesIO(image_data))   # initialize model
        image = image.convert("RGB")
        print(image.size)
        print(image.mode)
        model = load_model("/Users/sid/Desktop/Code/shloka-flask/converted_keras/keras_model.h5", compile=False)

        # Load the labels
        class_names = open("/Users/sid/Desktop/Code/shloka-flask/converted_keras/labels.txt", "r").readlines()
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        return jsonify(result=class_name[2:], probability=str(confidence_score))


if __name__ == '__main__':
    app.run(port=5002)
    app.run(debug=True)
    app.run()