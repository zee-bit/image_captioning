# Importing utility python libraries
import os
import urllib.request
from werkzeug.utils import secure_filename
import numpy as np

# Importing flask libraries
from flask import Flask, flash, request, redirect, render_template, url_for

# Importing ML libraries
import keras
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

# Importing caption generator function
from predict_main import generate_caption

# Setting session instances
graph = tf.get_default_graph()
session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

# Initializing flask application
app = Flask(__name__)

# Configuring flask settings
UPLOAD_FOLDER = "static/uploads/"
app.secret_key = "secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])

# Function for checking uploaded file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Setting route for handling 'GET' requests
@app.route("/")
def index():
    return render_template("index.html")


# Setting route for handling 'POST' requests
@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return redirect(request.url)
    else:
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            filepath = "./static/uploads/" + filename
            global graph
            with graph.as_default():
                set_session(session)
                caption = generate_caption(filepath)
            print("***" + caption + "***")
            return render_template("index.html", filename=filepath, caption=caption)
        else:
            flash("Allowed image types are : .png, .jpg, .jpeg")
            return redirect(request.url)


if __name__ == "__main__":
    app.run()
