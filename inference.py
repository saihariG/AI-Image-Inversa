import pickle
import numpy as np 
import tensorflow as tf
import config as cfg
import cv2

from utils.utils import *
from utils.dataset import image_loader

from model import ImageSearchModel

def simple_inference(model,
                     session,
                     train_set_vectors,
                     uploaded_image_path,
                     image_size,
                     distance = 'hamming'):
    
    
    '''
    Doing simple inference for single uploaded image
    
    :param model: CNN model
    :param session: tf.Session, restored session
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: string, path to the uploaded image
    :param image_size: tuple, single image (height, width)
    :param distance: string, type of distance to be used, 
                             this parameter is used to choose a way how to prepare vectors
    
    '''
    
    image = image_loader(uploaded_image_path, image_size)
    
    feed_dict = {model.inputs:[image], model.dropout_rate:0.0}
    
    dense_2_features , dense_4_features = session.run([model.dense_2_features, model.dense_4_features], feed_dict = feed_dict)
    
    closest_ids = None
    
    if distance == "hamming":
        
        dense_2_features = np.where(dense_2_features < 0.5 , 0 ,1)
        dense_4_features = np.where(dense_4_features < 0.5 , 0 ,1)                          
        
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)
        
    elif distance == "cosine":
        
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)
        
    return closest_ids

def compare_color(color_vectors,
                  uploaded_image_colors,
                  ids):
    '''
    Comparing color vectors of closest images from the training set with a color vector of a uploaded image (query image).
    
    :param color_vectors: color features vectors of closest training set images to the uploaded image
    :param uploaded_image_colors: color vector of the uploaded image
    :param ids: indices of training images being closest to the uploaded image (output from a distance function)
    '''
    color_distances = []
    
    for i in range(len(color_vectors)):
        color_distances.append(euclidean(color_vectors[i], uploaded_image_colors))
        
    #The 15 is just an random number that I have choosen, you can return as many as you need/want
    return ids[np.argsort(color_distances)[:15]]

def simple_inference_with_color_filters(model,
                                         session,
                                         train_set_vectors,
                                         uploaded_image_path,
                                         color_vectors,
                                         image_size,
                                         distance='hamming'):
    
    '''
    Doing simple inference for single uploaded image.
    
    :param model: CNN model
    :param session: tf.Session, restored session
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: string, path to the uploaded image
    :param color_vectors: loaded training set color features vectors
    :param image_size: tuple, single image (height, width)
    :param distance: string, type of distance to be used,
                             this parameter is used to choose a way how to prepare vectors
    '''
    
    image = image_loader(uploaded_image_path, image_size)
    
    ####################################################
    ## Calculating color histogram of the query image ##
    channels = cv2.split(image)
    features = []
    for chan in channels:
             hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
             features.append(hist)
 
    color_features = np.vstack(features).T
    ####################################################
    
    feed_dict = {model.inputs:[image], model.dropout_rate:0.0}
    
    dense_2_features, dense_4_features = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)
    
    closest_ids = None
    if distance == 'hamming':
            dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
            dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
            uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
            closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)
        
            #Comparing color features between query image and closest images selected by the model
            #################################################
            ## Compare color vectors ########################
            closest_ids = compare_color(np.array(color_vectors)[closest_ids], color_features, closest_ids)
        
    elif distance == 'cosine':
             uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
             closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)
        
             #Comparing color features between query image and closest images selected by the model
             #################################################
             ## Compare color vectors ########################
             closest_ids = compare_color(np.array(color_vectors)[closest_ids], color_features, closest_ids)
        
    return closest_ids

