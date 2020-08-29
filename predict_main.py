# %%

# Importing libraries for image manipulation, deep-learning and pickling
from pickle import load
import tensorflow as tf

# Importing functionalities from 'keras' library
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

# %%


def generate_caption(filepath):

    # Importing custom modules

    from utils import image_processing
    from utils import caption_writer

    # Loading files to map predicted indices to words and vice-versa

    ixtoword = load(open("./resources/ixtoword.pkl", "rb"))
    wordtoix = load(open("./resources/wordtoix.pkl", "rb"))

    # Initializing and customizing the InceptionV3 Model

    model = InceptionV3(weights="imagenet")
    model_new = Model(model.input, model.layers[-2].output)

    # Loading pre-trained weights and compiling the model

    nlp_model = tf.keras.models.load_model("model_weights/final_model.h5")
    nlp_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
    )

    # Setting maximum length of caption
    max_length = 72

    # Initializing empty dictionary for encoding custom images
    new_encoding_test = {}

    # Selecting and processing single test image into the encoding dictionary

    new_encoding_test[0] = image_processing.encode(filepath, model_new)

    # Reshaping encoded image and generating it's captions using 'nlp_model'

    image = new_encoding_test[0].reshape((1, 2048))
    caption = caption_writer.greedySearch(
        image, max_length, wordtoix, ixtoword, nlp_model
    )

    # Formatting the raw captions

    final_caption = caption_writer.caption_format(caption)

    # Returning the final caption

    return final_caption
