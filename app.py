# some utilities
import os
import numpy as np
from util import base64_to_pil,upload_image
import base64
from PIL import Image,ImageOps
from io import BytesIO
# Flask
from pydub import AudioSegment
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from flask import send_file
import requests

#tensorflow
# import tensorflow as tf
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

#mediaPipe
# from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from flask import Flask, request, jsonify
import io
# import tensorflow as tf
from PIL import Image

# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
# Model_json = ".json"
# Model_weigths = ".h5"


from flask_cors import CORS

# Declare a flask app
app = Flask(__name__)
CORS(app)

# model = load_model("keras_model.h5", compile=False)
# # # Load the labels
# class_names = open("labels.txt", "r").readlines()


#mediaPipe
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


# model_p = load_model("keras_model_p.h5", compile=False)
# # # Load the labels
# class_names_p = open("labels_p.txt", "r").readlines()


import openai
import os

# from boto.s3.connection import S3Connection

@app.route('/process_wav_js', methods=['POST'])
def process_wav_js():
    file = request.files['audio']
    file.save("audio.wav")
    openai.api_key = os.environ['transcription_key'] 
    print(openai.api_key)
    audio_file= open("audio.wav", "rb")
    result = openai.Audio.transcribe("whisper-1", audio_file)
    print('here is the audio text:', result["text"])
    return jsonify(result["text"])

@app.route('/process_wav_js_key', methods=['POST'])
def process_wav_js_key():
    file = request.files['audio']
    api_key = request.form.get('key')
    file.save("audio.wav")
    openai.api_key = api_key #os.environ['transcription_key'] 
    print(openai.api_key)
    audio_file= open("audio.wav", "rb")
    result = openai.Audio.transcribe("whisper-1", audio_file)
    print('here is the audio text:', result["text"])
    return jsonify(result["text"])


@app.route('/process_wav', methods=['POST'])
def process_wav():
    print('request')
    audio = request.get_data()  
    if not audio:
        return 'No audio data in the request', 400
    wav_path = 'audio.wav'
    with open(wav_path, 'wb') as f:
        f.write(audio)

    openai.api_key = os.environ['transcription_key'] #"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" #S3Connection(os.environ['transcription_key'])
    print(openai.api_key)
    audio_file= open(wav_path, "rb")
    result = openai.Audio.transcribe("whisper-1", audio_file)
    print('here is the audio text:', result["text"])
    return jsonify(result["text"])


@app.route("/upload", methods = ["GET", "POST"])
def upload():
    print(os.environ['transcription_key'])
    return jsonify(os.environ['transcription_key'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    print('Gesture Hit')
    if request.method == 'POST':
        image_data = request.files['image'].read()

        # Convert the image data to a PIL Image object
        image = Image.open(io.BytesIO(image_data))   
        width, height = image.size
        midpoint = width // 2

        # Split the image
        right_half = image.crop((midpoint, 0, width, height))
        # cut_width = int(0.2 * right_half.size[0])
        # cut_height = int(0.2 * right_half.size[1])
        cut_width_left = int(0.2 * right_half.size[0])
        cut_height_top = int(0.2 * right_half.size[1])
        # Adjust the width for the left cut
        new_width = right_half.size[0] - cut_width_left

        # Crop the right half image
        cropped_right_half = right_half.crop((cut_width_left, cut_height_top, new_width, right_half.size[1]))

        
        # Crop the right half image
        cropped_right_half = right_half.crop((cut_width, cut_height, right_half.size[0], right_half.size[1]))

        #right_half.save('right_half.jpg')# initialize model
        cropped_right_half.save("img1.png")
        image = mp.Image.create_from_file('img1.png')
        # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(image)
        print(recognition_result)
        try:
            top_gesture = recognition_result.gestures[0][0]
            print(top_gesture.category_name)

            # return jsonify(result=class_name[2:], probability=str(confidence_score))
            return jsonify(result = top_gesture.category_name)
        except:
            # return jsonify(result=class_name[2:], probability=str(confidence_score))
            return jsonify(result = "Nothing Recognized")




@app.route('/get_image')
def get_image():
    print('hit')
    return send_file('img1.png', mimetype='image/png')

# @app.route('/predictPrana', methods=['GET', 'POST'])
# def predictPrana():
#     '''
#     predict function to predict the image
#     Api hits this function when someone clicks submit.
#     '''
#     if request.method == 'POST':
#       # Get the image data from the POST request
#         image_data = request.files['image'].read()

#         # Convert the image data to a PIL Image object
#         image = Image.open(io.BytesIO(image_data))   # initialize model
#         image = image.convert("RGB")
#         print(image.size)
#         print(image.mode)
#         data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#         # resizing the image to be at least 224x224 and then cropping from the center
#         size = (224, 224)
#         image = ImageOps.fit(image, size)

#         # turn the image into a numpy array
#         image_array = np.asarray(image)

#         # Normalize the image
#         normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

#         # Load the image into the array
#         data[0] = normalized_image_array

#         # Predicts the model
#         prediction = model_p.predict(data)
#         index = np.argmax(prediction)
#         class_name = class_names_p[index]
#         confidence_score = prediction[0][index]

#         # Print prediction and confidence score
#         print("Class:", class_name[2:], end="")
#         print("Confidence Score:", confidence_score)

#         return jsonify(result=class_name[2:], probability=str(confidence_score))




if __name__ == '__main__':
    app.run(port=5002)
    app.run(debug=True)
    app.run()