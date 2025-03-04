import os
import subprocess
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from tensorflow.keras.utils import get_custom_objects

app = Flask(__name__)

GITHUB_REPO = "https://github.com/Chaeun26/flask-deploy.git"
MODEL_DIR = "flask-deploy"
MODEL_PATH = "elmo_model.h5"

# to download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):  # If model doesn't exist
        print("Downloading model from GitHub...")
        if not os.path.exists(MODEL_DIR):
            subprocess.run(["git", "clone", GITHUB_REPO], check=True)
        if os.path.exists(f"{MODEL_DIR}/{MODEL_PATH}"):
            subprocess.run(["cp", f"{MODEL_DIR}/{MODEL_PATH}", "."], check=True)
        print("Model downloaded successfully.")

# Run this before loading the model
download_model()


def elmo_embedding(x):
    elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=False)
    return elmo(tf.squeeze(x, axis=1))

get_custom_objects().update({"elmo_embedding":elmo_embedding})

# Load the model with custom objects
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"elmo_embedding": elmo_embedding})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = tf.convert_to_tensor([data['text']], dtype=tf.string)
    prediction = model.predict(input_text)[0][0]
    prediction = round(float(prediction),4)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
