import tensorflow as tf
import tensorflow.compat.v1 as tf1
#tf.compat.v1.disable_eager_execution()
tf1.disable_eager_execution()

def model_inputs(image_size):
    '''
    Defines CNN inputs (placeholders)
    
    :param image_size : tuple , (height , width) of an  image
    '''
    # -> [Batch_size , image_size[0] , image_size[1] , 3]
    inputs  = tf1.placeholder(dtype = tf.float32 , shape=[None , image_size[0] , image_size[1] , 3] , name='images')
    
    targets = tf1.placeholder(dtype = tf.int32 , shape = [None] , name='targets')
    dropout_rate = tf1.placeholder(dtype = tf.float32 , name='dropout_rate')
    
    
    return inputs , targets , dropout_rate

def conv_block(inputs , 
               number_of_filters,
               kernel_size ,
               strides = (1,1),
               padding = 'SAME',
               activation = tf.nn.relu ,
               max_pool = True ,
               batch_norm = True ):
    
    
    '''
    Defines Convolutional block layer
    
    :param inputs: data from a previous layer
    :param number_0f_filters : integer , number of conv filters
    :param kernel_size : tuple , size of conv layer kernel
    :param padding : string , type of padding technique : SAME or VALID
    :param activation : tf.object , activation function used on layer 
    :param max_pool : boolean , if true the conv block will use max_pool
    :param batch_norm : boolean , if true the conv block will use batch normalization
      
    '''
    
    conv_features = layer  = tf1.layers.conv2d( inputs = inputs , 
                                               filters = number_of_filters, 
                                               kernel_size = kernel_size ,
                                               strides = strides ,
                                               padding  = padding ,
                                               activation = activation 
                                             )
    
    
    if max_pool:
        
        layer = tf1.layers.max_pooling2d(layer , pool_size = (2,2) , strides=(2, 2) , padding = 'SAME' )
        
    if batch_norm:
        layer = tf1.layers.batch_normalization(layer)
        
    return layer , conv_features

def dense_block(inputs,
                units,
                activation = tf.nn.relu ,
                dropout_rate = None ,
                batch_norm = True):
    '''
    Defines dense block layer
    
    :param inputs : data from a previous layer
    :param units : Integer , number of neurons/units for a dense layer
    :param activation : tf.object , activation function used on layer 
    :param dropout_rate : dropout rate used in this dense block
    :param batch_norm : boolean , if true the conv block will use batch normalization
    
    '''
    dense_features = layer = tf1.layers.dense(inputs = inputs ,
                                             units = units ,
                                             activation = activation )
    
    if dropout_rate is not None:
        layer = tf1.layers.dropout(layer , rate = dropout_rate )
        
    if batch_norm :
        layer = tf1.layers.batch_normalization(layer)
        
    return layer , dense_features

def opt_loss(logits , targets , learning_rate):
    '''
    Define model's optimizer and loss function
    
    :param logits: pre-activated model outputs
    :param targets : true labels for each input sample
    :param learning_rate : learning_rate
    '''
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets , logits = logits))
    
    optimizer = tf1.train.AdamOptimizer(learning_rate).minimize(loss)
    
    #tf.optimizers.Adam
    #tf.train.AdamOptimizer
    
    return loss , optimizer