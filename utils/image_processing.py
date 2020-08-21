import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

#%%

# Image-Preprocessing Function : takes path of the image as the only argument
def preprocess(image_path):

    # Loading Image file
    img = tf.keras.preprocessing.image.load_img(image_path)

    # Resizing image to 299x299px for feeding it into InceptionV3 Model
    img = img.resize((299, 299), resample=0)

    # Converting image into numpy array and returning the array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


#%%

# Image-Encoding Function : takes PIL image and the trained model as the two arguments
def encode(image, model_new):

    # Pre-processing image using the 'preprocess' function
    image = preprocess(image)

    # Creating, reshaping and returning the numpy feature-vector of the image
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec
