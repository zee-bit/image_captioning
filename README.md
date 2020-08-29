# Image Caption Generator

## Overview

This is a deep learning-based image captioning project. It allows the user to upload an image through a webpage trained VGG16 model to generate and show its captions. The application accepts .jpg, .jpeg and .png format images and generates captions of up to 35 words.

## Application Structure

### FRONT END

The front-end web application is present in 'app.py' file and is written using the flask library in Python. It allows the user to access the webpage in their local server and provides a layout for uploading images and viewing captions.

### PREDICT FILE

The 'predict_main.py' file contains the necessary functions to generate the final caption and format it to make it more presentable. It inherits both the utility modules 'caption_writer.py' and 'image_processing.py' stored in the 'utils' folder. It offers the 'generate_caption' function which is inherited by the 'app.py' file.

### TRAINING FILE

The 'training_main.py' file is the python file in which the training of the model is being done. The training images are used from the 'flick-30k' dataset available on Kaggle. It imports the utility modules as well as three training modules stored in the 'training_modules' folder. The images are reduced to the feature-vector form using the VGG16 model. The reduced feature-vectors are then trained to learn the captions using LSTM models and the model is pickled in the 'model_weights' folder as 'final_model.h5' file.

### UTILITIES

1. **'caption_writer.py'**: It provides the necessary functions for mapping the caption words from the index-to-word and word-to-index dictionaries which it accepts as input and outputs the unformatted caption as a raw string. It also contains the function to add punctuations, enforce grammatical standards and remove repetitions from the generated caption.

2. **'image_processing.py**: It encodes the input image to the form which is compliant with the model and also reduces into its feature-vector format. The output image dimensions are 300x300 px.

### TRAINING MODULES

1. **'description_processor.py'**: It imports and cleans the description of the flickr-30k image dataset and provides the model compliant descriptions as output in the form of a mapped dictionary.

2. **'description_properties.py'**: It provides the various parameters for the model output such as the maximum length of the function and the common wordset which is to be used, thus providing the model with the caption-vocabulary to work with.

3. **'training_functions.py'**: It contains functions to import and tune the 'glove' wordset as well as initializes and configures the embedding matrix and mapping dictionaries. It provides the 'data_generator' function which takes the image and model parameters as input and provides the model with appropriate input matrices.

## Screenshots

### Empty User-Interface

![Empty User-interface](https://i.imgur.com/piDA0RT.png)

### Sample Caption

![Sample Caption](https://i.imgur.com/sD83QJ0.png)
