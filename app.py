import os
import subprocess
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define GitHub repository and model path
GITHUB_REPO = "https://github.com/Chaeun26/flask-deploy.git"
MODEL_PATH = "elmo_model.h5"

# Function to download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GitHub...")
        if not os.path.exists("flask-deploy"):
            subprocess.run(["git", "clone", GITHUB_REPO], check=True)
        subprocess.run(["mv", "flask-deploy/elmo_model.h5", "."], check=True)
        print("Model downloaded successfully.")

# Download the model before starting the app
download_model()

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Preprocess input text
    input_text = tf.convert_to_tensor([text], dtype=tf.string)

    # Make prediction
    prediction = model.predict(input_text)[0][0]

    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
