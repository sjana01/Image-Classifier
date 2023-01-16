# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:05:20 2022

@author: Santanil Jana
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
model_path = 'resnet50.h5'

# Load Model
model = load_model(model_path)
model.make_predict_function()       # Necessary, can also built custom functions 

# Preprocessing the images and predict using the model
def model_predict(img_path, model):
    # input shape has to be always (224, 224, 3) except when include_top is False 
    img = image.load_img(img_path, target_size=(224,224))
    
    # preprocessing the image
    x = image.img_to_array(img)
    # expand the dimensions of the image for the model
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    
    # predictions
    preds = model.predict(x)
    
    return preds


# app route. A GET message is send, and the server returns data
@app.route('/', methods=['GET'])
def index():
    # The main page. render_template is required to show the homepage
    # This is kind of an UI that will enable us to upload images 
    return render_template('index.html')


#Used to send HTML form data to the server. 
#The data received by the POST method is not cached by the server.
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process the result 
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None
        


if __name__ == '__main__':
    app.run(debug=True)