# Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np 
# from flask import Flask , redirect , url_for , request , render_template
# import os
# import glob
# from werkzeug.utils import secure_filename


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cv2
from cv2 import imread, createCLAHE
from tqdm import tqdm
import themodel

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image
import numpy as np
from PIL import Image as im
import numpy as np
from skimage import transform

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model_path = 'model_name.h5'
model = load_model(model_path)
# model._make_predict_function()  # Necessary


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (512, 512, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def predictions(img_path , model):
    img =image.load_img(img_path , target_size=(512,512))
    #img.reshape(1,128,128,3)

   # x=image.img_to_array(img)
    #x=x/255
   # x=np.expand_dims(x,axis=0)
    #pred = np.argmax(model.predict(img)[0], axis=-1)
    pred_img2 = model.predict(image)

    if pred_img2 > 0.5:
        preds="yes"
    else:
        preds="no"


    return preds



@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        img = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath , 'uploads' , secure_filename(img.filename))
        img.save(file_path)

        result = predictions(file_path , model) # return index
        

        return result
    return None 







app.run(debug=True)
