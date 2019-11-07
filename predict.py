import keras
import tensorflow as tf
from keras.models import Sequential
from keras.utils import np_utils
import os
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image

batch_size = 32  # orig paper trained all networks with batch_size=128
input_shape = (32,32,3) # input shape of image
baseMapNum = 32 # number of units
weight_decay = 1e-4
num_classes = 10 # number of classes

#added to work with flask
graph = tf.Graph() 
with graph.as_default():
    session = tf.Session()
    with session.as_default():
     
        #building CNN model 
        model = Sequential()
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        model.compile()
        model.load_weights("./saved_models/cifar10_cnn2.h5")
        model._make_predict_function()

def run(data):
    
    image_data = re.sub('^data:image/.+;base64,', '', data) # remove header in post data
    image_data = base64.b64decode(image_data) # decode base64 value 
    image = Image.open(BytesIO(image_data)) # import to image
    image = image.resize((32,32)) # resize to fit cnn
    np_image = np.array(image) # change to np_array
    np_image = np_image.astype('float32')/255 #normalize
    np_image = np.array([np_image]) # add dimension: (1,32,32,3)
    
    classes = np.array(['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ships', 'truck']) #classes
    
    #added to work with flask
    with graph.as_default():
        with session.as_default():
            result = model.predict(np_image) #predict image 
    return classes[(result>0.5)[0]][0] # to change from categorical to string value