# %%

# Importing generic python libraries
import string
import os
import math
import glob
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time

# Importing libraries for image manipulation, deep-learning and pickling
from PIL import Image, ImageOps
from pickle import dump, load
import tensorflow as tf

# Importing functionalities from 'keras' library
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# %%

# Importing custom modules

from utils import image_processing
from utils import caption_writer


# %%

# Loading files to map predicted indices to words and vice-versa

ixtoword=load(open('./resources/ixtoword.pkl','rb'))
wordtoix=load(open('./resources/wordtoix.pkl','rb'))

# Setting maximum length of caption
max_length=72


# %%

# Initializing and customizing the InceptionV3 Model

model=InceptionV3(weights="imagenet")
model_new=Model(model.input,model.layers[-2].output)

# Loading pre-trained weights and compiling the model

nlp_model=tf.keras.models.load_model('model_weights/final_model.h5')
nlp_model.compile(loss='categorical_crossentropy',optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001))


# %%

# Initializing the path of the test images

test_folder='./test/'
test_images=glob.glob(test_folder+'*.jpg')


# %%

# Setting time-stamp as current time and initializing empty dictionary 'encoding_test'

start=time()
encoding_test={}

# Looping through and encoding all the images in the test directory

for img in test_images:
    encoding_test[img[len(test_folder):]]=image_processing.encode(img, model_new)
    #print("time taken in second=",time()-start)


# %%

# Listing the keys of all the test images
pics=list(encoding_test.keys())

# Looping through all the mapped images of the test directory
for pic in pics:
    
    # Reshaping the encoding ndarray
    image=encoding_test[pic].reshape((1,2048))
    
    # Generating and printing captions for the encoded image using the trained 'nlp_model'
    print(caption_writer.greedySearch(image, max_length, wordtoix, ixtoword, nlp_model))
    print('\n')


# %%

# Initializing empty dictionary for encoding custom images
new_encoding_test={}

# Selecting and processing single test image and mapping it into the encoding dictionary

pic_name='1440465.jpg'
new_encoding_test[pic_name]=image_processing.encode(pic_name, model_new)

# Reshaping encoded image and generating it's captions using 'nlp_model'

image=new_encoding_test[pic_name].reshape((1,2048))
print(caption_writer.greedySearch(image, max_length, wordtoix, ixtoword, nlp_model))

