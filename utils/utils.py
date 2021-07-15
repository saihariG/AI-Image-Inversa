import numpy as np
from scipy.spatial.distance import hamming, cosine, euclidean

def cosine_distance(training_set_vectors , query_vector , top_n = 50):
    '''
    Calculates cosine distance b/w query image (vector) and all training set images (vectors)
    
    :param training_set_vectors: numpy Matrix , vectors for all images in the training set
    :param query_vector : numpy Vector , query image (new image) vector
    :param top_n : integer , number of closest images to return 
    
    '''
    distances = []
    
    for i in range(len(training_set_vectors)): # for cifar-10 Dataset -> 50k images
        distances.append(cosine(training_set_vectors[i] , query_vector[0]))
    
    return np.argsort(distances)[:top_n]   

def hamming_distance(training_set_vectors , query_vector , top_n = 50):
    '''
     Calculates hamming distances b/w query image (vector) and all training set images (vectors)
    
    :param training_set_vectors: numpy Matrix , vectors for all images in the training set
    :param query_vector : numpy Vector , query image (new image) vector
    :param top_n : integer , number of closest images to return 
    
    '''
    distances = []
    
    for i in range(len(training_set_vectors)): # for cifar-10 Dataset -> 50k images
        distances.append(hamming(training_set_vectors[i] , query_vector[0]))
    
    return np.argsort(distances)[:top_n]  

def sparse_accuracy(true_labels , predicted_labels):
    '''
    Calculates accuracy of a model based on softmax outputs
    
    :param true_labels : numpy array . real labels of each sample. eg: [1,2,1,0,0]
    :param predicted_labels : numpy matrix , softmax probabilities. Eg. [[02 , 0.1, 0.7] , [0.9 , 0.05 , 0.05]]
    '''
    
    assert len(true_labels) == len(predicted_labels)
    
    correct  = 0
    
    for i in range(len(true_labels)):
        
        if np.argmax(predicted_labels[i]) == true_labels[i]:
            
            correct += 1
            
    return correct / len(true_labels)