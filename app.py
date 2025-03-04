from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load ELMo from TensorFlow Hub
elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=False)

def elmo_embedding(x):
    x = tf.reshape(x, [-1])  # Ensure input is a 1D tensor
    return elmo(x)

# Load the .h5 model with the custom ELMo embedding function
model = tf.keras.models.load_model("elmo_model.h5", custom_objects={"elmo_embedding": elmo_embedding})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = [data["text"]]  # Convert to list format
    input_tensor = tf.convert_to_tensor(input_text, dtype=tf.string)
    prediction = model.predict(input_tensor)[0][0]

    # Format response
    result = round(float(prediction), 4)  # Round for better readability
    return jsonify({"prediction": result, "confidence": f"{result * 100:.2f}%"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
