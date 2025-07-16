from flask import Flask, request, jsonify
import torch
from lmu_model import LMU
import pandas as pd
from preprocess import load_data, preprocess_data

app = Flask(__name__)

# Load and preprocess data
texts, emotions = load_data('dataset.csv')
texts, emotions = preprocess_data(texts, emotions)

# Assume you have trained your model and saved it as 'lmu_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LMU(input_size=embedding_dim, hidden_size=hidden_size, output_size=num_classes)
model.load_state_dict(torch.load('lmu_model.pth', map_location=device, weights_only=True))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']

    # Tokenize and prepare your input text here
    input_tensor = ...  # Convert input text to tensor format

    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to a readable format
    predicted_emotion = torch.argmax(output, dim=1).item()
    return jsonify({'emotion': predicted_emotion})


if __name__ == '__main__':
    app.run(debug=True)
