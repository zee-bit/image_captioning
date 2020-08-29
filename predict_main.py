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

    # Splitting the caption into words and resizing the length

    f_caption = caption.split()
    f_caption = f_caption[0:35]

    # Removing consecutive replications

    for i in range(35):
        if i + 1 < len(f_caption):
            if f_caption[i] == f_caption[i + 1]:
                del f_caption[i + 1]
    # Adding comma punctuation

    for i in range(35):
        if i + 2 < len(f_caption):
            if f_caption[i] == "and" and f_caption[i + 2] == "and":
                f_caption[i] = ","
    # Removing repitition of multi-word caption terms

    i = 0
    while i + 3 < len(f_caption):
        if f_caption[i] == f_caption[i + 2]:
            if f_caption[i + 1] == f_caption[i + 3]:
                f_caption[i + 2] = f_caption[i + 3] = "$"
        i += 1
    f_caption = list(filter(lambda a: a != "$", f_caption))

    i = 0
    while i + 5 < len(f_caption):
        if f_caption[i] == f_caption[i + 3]:
            if f_caption[i + 1] == f_caption[i + 4]:
                if f_caption[i + 2] == f_caption[i + 5]:
                    f_caption[i + 3] = f_caption[i + 4] = "$"
                    f_caption[i + 5] = "$"
                    f_caption = list(filter(lambda a: a != "$", f_caption))
                    i -= 3
        i += 1
    # Removing blank space adjacent to punctuations

    i = 0
    while i < len(f_caption):
        if f_caption[i] == ",":
            f_caption[i - 1] = f_caption[i - 1] + ","
            del f_caption[i]
            i -= 1
        i += 1
    # Removing the occurrence of auxiliary words as the last term

    aux_words = ["and", "or", "is", "was", "a", "an", "it", "of", "the"]
    end_word = f_caption[-1]
    if end_word in aux_words:
        del f_caption[-1]
    # Rejoining the caption and adding sentence formatting

    f_caption = " ".join(f_caption)
    f_caption = f_caption[0].upper() + f_caption[1:] + "."

    # Returning the formatted captions

    return f_caption
