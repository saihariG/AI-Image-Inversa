import tensorflow as tf
import tensorflow.compat.v1 as tf1
from utils.model import *

class ImageSearchModel(object):
    
    def __init__(self ,
                 learning_rate ,
                 image_size ,
                 number_of_classes = 10 ): 
        '''
        Define CNN Model 
        
        :param learning_rate : learning_rate
        :param image_size : tuple , (height , width) of an image
        :param number_of_classes : integer , number of classes in dataset
        '''
        
        #tf.reset_default_graph()
        
        tf.compat.v1.reset_default_graph()
        #ops.reset_default_graph()
        
        self.inputs , self.targets , self.dropout_rate = model_inputs(image_size)
        
        normalized_images = tf1.layers.batch_normalization(self.inputs)
        
        #conv block 1
        
        conv_block_1 , self.conv_1_features = conv_block(inputs = normalized_images,
                                                         number_of_filters = 64 , 
                                                         kernel_size = (3,3) , 
                                                         strides=(1,1) , 
                                                         padding = 'SAME' ,
                                                         activation = tf.nn.relu ,
                                                         max_pool = True ,
                                                         batch_norm = True) 
        
        
        #conv block 2
        
        conv_block_2 , self.conv_2_features = conv_block(inputs = conv_block_1 ,
                                                         number_of_filters = 128 , 
                                                         kernel_size = (3,3) , 
                                                         strides=(1,1) , 
                                                         padding = 'SAME' ,
                                                         activation = tf.nn.relu ,
                                                         max_pool = True ,
                                                         batch_norm = True) 
        
        
        #conv block 3
        
        conv_block_3 , self.conv_3_features = conv_block(inputs = conv_block_2 ,
                                                         number_of_filters = 256 , 
                                                         kernel_size = (5,5) , 
                                                         strides=(1,1) , 
                                                         padding = 'SAME' ,
                                                         activation = tf.nn.relu ,
                                                         max_pool = True ,
                                                         batch_norm = True) 
        
        #conv block 4
        
        conv_block_4 , self.conv_4_features = conv_block(inputs = conv_block_3 ,
                                                         number_of_filters = 512 , 
                                                         kernel_size = (5,5) , 
                                                         strides=(1,1) , 
                                                         padding = 'SAME' ,
                                                         activation = tf.nn.relu ,
                                                         max_pool = True ,
                                                         batch_norm = True) 
        
        
        
        #flattening 
        flat_layer  = tf1.layers.flatten(conv_block_4)
        
        
        # Dense block 1
        
        dense_block_1 , dense_1_features = dense_block(inputs = flat_layer ,
                                                       units = 128 ,
                                                       activation = tf.nn.relu ,
                                                       dropout_rate = self.dropout_rate ,
                                                       batch_norm = True )
        
        
        
        
        
        # Dense block 2
        
        dense_block_2 , self.dense_2_features = dense_block(inputs = dense_block_1 ,
                                                       units = 256 ,
                                                       activation = tf.nn.relu ,
                                                       dropout_rate = self.dropout_rate ,
                                                       batch_norm = True )
        
        
        
        # Dense block 3
        
        dense_block_3 , self.dense_3_features = dense_block(inputs = dense_block_2 ,
                                                       units = 512 ,
                                                       activation = tf.nn.relu ,
                                                       dropout_rate = self.dropout_rate ,
                                                       batch_norm = True )
        
        
        
           
        # Dense block 4
        
        dense_block_4 , self.dense_4_features = dense_block(inputs = dense_block_3 ,
                                                       units = 1024 ,
                                                       activation = tf.nn.relu ,
                                                       dropout_rate = self.dropout_rate ,
                                                       batch_norm = True )
        
        
        
        
        logits = tf1.layers.dense(inputs = dense_block_4 , 
                                 units = number_of_classes , 
                                 activation = None )
        
        
        self.predictions = tf.nn.softmax(logits)
        
        self.loss , self.opt = opt_loss(logits = logits  ,
                                        targets = self.targets ,
                                        learning_rate = learning_rate)