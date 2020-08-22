# %%

# Importing generic python libraries
import glob
from time import time

# Importing libraries for image manipulation, deep-learning and pickling
from pickle import dump, load
import tensorflow as tf

# Importing functionalities from 'keras' library
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import Input

# %%

# Importing custom modules

from utils import image_processing
from training_modules import description_processor
from training_modules import description_properties
from training_modules import training_functions

# %%

# Loading processed training descriptions into 'descriptions' dictionary

descriptions = description_processor.load_final_descriptions()

# %%

# Printing Sample-keys and Sample-descriptions

print("Sample keys : ", list(descriptions.keys())[:5], "\n")
print("Sample Description 1 : ", descriptions["1000268201"], "\n")
print("Sample Description 2 : ", descriptions["1000344755"], "\n")

# %%

# Adding all distinct words in the description dictionary to the 'all_desc' set

all_desc = set()
for key in descriptions.keys():
    # print(key)
    for d in descriptions[key]:
        [all_desc.update(d.split())]
# %%

# Printing the number of distinct words present in the descriptions

vocabulary = all_desc
print("size of the vocabulary=", len(vocabulary))

# %%

# Loading and printing the size of the description dataset

filename = "results.csv"
train = description_processor.load_set(filename)
print("dataset=", len(train))

# %%

# Initializing the images-dataset directory
# Adding the name of each image into 'img' list

images = "./flickr30k_images/flickr30k_images/"
img = glob.glob(images + "*.jpg")

# %%

# Initializing and customizing the InceptionV3 model

model = InceptionV3(weights="imagenet")
model_new = Model(model.input, model.layers[-2].output)

# %%

# Setting time-stamp as current time
# Initializing empty dictionary 'encoding_train'
start = time()
encoding_train = {}

# Looping through and encoding all the images in the train directory

for img in img:
    encoding_train[img[len(images):]] = image_processing.encode(img)
    print("Time taken in second=", time() - start)
# %%

# Pickling the encoded training images dataset

with open("./resources/encoded_train_images.pkl", "wb") as f:
    dump(encoding_train, f)
# %%

# Loading the pickled encoded-image dataset
# Initializing it as the training feature matrix

train_features = load(open("./resources/encoded_train_images.pkl", "rb"))
print(len(train_features))

# %%

# Adding all the training captions to a list and printing it's length

all_train_captions = []
for key, val in descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

# %%

# Prints the word count of the description dataset
# Iinitializes the frequent-words list

vocab = description_properties.description_vocabulary(all_train_captions)

# %%

# Saving locally the dictionaries mapping indices to words and vice-versa

(ixtoword, wordtoix) = training_functions.get_mapping_dicts(vocab)

# %%

# Printing the size of the vocabulary using the index-to-word mapping dict
vocab_size = len(ixtoword) + 1
print(vocab_size)

# %%

# Prints the number of lines of descriptions in the training set

print(len(description_properties.to_lines(descriptions)))

# %%

# Prints the length of the longest description in the description dataset

max_length = description_properties.max_length(descriptions)
print("Length of max-length description = ", max_length)

# %%

# Making a matrix of all words common in the glove set and dict

embedding_dim = 200
embedding_matrix = training_functions.get_embedding_matrix(
    embedding_dim, wordtoix, vocab_size
)

# %%

# Customizing the layers of the model
# Selecting the appropriate activation functions

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)
inputs2 = Input(shape=(max_length,))

se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# %%

# Printing the summary of the model

model.summary()

# %%

# Setting the weights of the model layer

model.layers[2]
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

# %%

# Compiling the model (selecting loss function and 'adam' optimizer)

model.compile(loss="categorical_crossentropy", optimizer="adam")

# %%

# Setting parameters for model training/optimization

epochs = 9
ppb = 3
steps = len(descriptions)

# %%

del descriptions["image_name"]
del descriptions[""]

# %%

# Saving the model(first instance) locally
model.save("./model_weights/model_" + str(0) + ".h5")

# %%

# Optimizing the model weighs (epoch number of times) and saving the weights

for i in range(epochs + 1):
    generator = training_functions.data_generator(
        descriptions, train_features, wordtoix, max_length, ppb, vocab_size,
    )
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("./model_weights/model_" + str(i) + ".h5")
# %%

# Storing the final trained model in 'new_model'


new_model = tf.keras.models.load_model("./model_weights/model_7.h5")

# %%

# Setting new parameters for model optimization

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
)
epochs = 10
number_pics_per_bath = 6
steps = len(descriptions) // number_pics_per_bath

# %%

# Optimizing the model weighs (epoch number of times)
# Saving the weights locally each time

for i in range(epochs + 1):
    generator = training_functions.data_generator(
        descriptions, train_features, wordtoix, max_length, ppb
    )
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("./model_weights/model_" + str(i + 10) + ".h5")
# %%

# Initializing final model with final-trained weights

model = tf.keras.models.load_model("./model_weights/final_model.h5")
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001),
)
