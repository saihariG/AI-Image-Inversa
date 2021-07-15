'''
Flask backend for image-to-image search pipeline
'''
# import dependencies

import os
import tensorflow as tf
import numpy as np
import pickle
import config as cfg

from model import ImageSearchModel
from inference import simple_inference , simple_inference_with_color_filters

# importing flask dependencies

from flask import Flask, request , render_template , send_from_directory

#set root dir

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#Define model

model = ImageSearchModel(learning_rate = cfg.LEARNING_RATE, image_size = cfg.IMAGE_SIZE ,
                         number_of_classes =  cfg.NUMBER_OF_CLASSES)

session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.Saver()
saver.restore(session , 'saver/model_epoch_5.ckpt') 

#Load training set vectors
with open('hamming_train_vectors.pickle', 'rb') as f:
	train_vectors = pickle.load(f)

#Load training set paths
with open('train_images_pickle.pickle', 'rb') as f:
	train_images_paths = pickle.load(f)

with open('color_vectors.pickle', 'rb') as f:
        color_vectors = pickle.load(f)

#define flask app
app = Flask(__name__ , static_url_path = '/static') #/static

#define apps home page
@app.route("/") #www.image-search.com/
def index():
    return render_template("index.html")

#Define upload function
@app.route("/upload", methods = ["POST"])
def upload():

    upload_dir = os.path.join(APP_ROOT,"uploads/")
     
    if not os.path.isdir(upload_dir):
            os.mkdir(upload_dir)
    
    for img in request.files.getlist("file"):
           img_name = img.filename
           destination = "/".join([upload_dir , img_name])
           img.save(destination)
      
    #inference
    result = np.array(train_images_paths)[simple_inference_with_color_filters(model , session , train_vectors,
                                          os.path.join(upload_dir, img_name), color_vectors ,cfg.IMAGE_SIZE)]      
            
    result_final = []

    # images/
    for img in result:
    	result_final.append("images/"+img.split("/")[-1]) #example: dataset/train/0_frog.png -> [dataset, train, 0_frog.png] -> [-1] = 0_frog.png  "images/"+img.split("/")[-1]   +img.split("/")[-1]
    
    return render_template("result.html", image_name=img_name, result_paths=result_final[:-2]) #added [:-2] just to have equal number of images in the result page per row              

# Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
        return send_from_directory("uploads" , filename)

#start the app


if __name__ == "__main__":
        app.run(port = 5000, debug=False, host = '0.0.0.0')









    


























 