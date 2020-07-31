import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

# Image-Preprocessing Function : takes path of the image as the only argument
def preprocess(image_path):
    
    #Loading Image file
    img=tf.keras.preprocessing.image.load_img(image_path)
    
    #Adding white padding to convert image into square dimensions
    size=img.size
    if(size[0]>size[1]):
        (size_max,size_min)=(size[0],size[1])
        border_dir='v'
    else:
        (size_max,size_min)=(size[1],size[0])
        border_dir='h'
    border_amount=math.ceil((size_max-size_min)/2)
    if (border_dir == 'v'):
        img = ImageOps.expand(img,border=(0,border_amount),fill='white')
    else:
        img = ImageOps.expand(img,border=(border_amount,0),fill='white')
    
    #Resizing image to 299x299px for feeding it into InceptionV3 Model
    img = img.resize((299,299),resample=0)
    
    #Converting image into numpy array and returning the array
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    return x
	

# Image-Encoding Function : takes PIL image and the trained model as the two arguments
def encode(image, model_new):
    
    #Pre-processing image using the 'preprocess' function
    image=preprocess(image)
    
    #Creating, reshaping and returning the numpy feature-vector of the image
    fea_vec=model_new.predict(image)
    fea_vec=np.reshape(fea_vec,fea_vec.shape[1])
    return fea_vec