import os
import urllib.request
import numpy as np
import keras
from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from predict_main import generate_caption

graph = tf.get_default_graph()

session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    else:
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = './static/uploads/' + filename
            global graph
            with graph.as_default():    
                set_session(session)
                caption = generate_caption(filepath)
            print("***" + caption + "***")
            return render_template('index.html', filename = filepath, caption = caption)

        else:
            flash("Allowed image types are : .png, .jpg, .jpeg")
            return redirect(request.url)

if __name__ == "__main__":
    app.run()