import pandas as pd
from datasets import load_dataset
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

dataset = load_dataset("jahjinx/IMDb_movie_reviews")

# Преобразуем в pandas DataFrame
df_train = pd.DataFrame(dataset['train'])


# Загрузка модели и токенизатора
model = load_model('./results/imdb_train_model.h5')
with open('./results/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Преобразование текста в последовательности
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=150)

    # Получение предсказания
    prediction = model.predict(padded_sequence)
    print(f'Prediction: {prediction}')  # Логирование предсказания

    # Преобразование результата в класс ("positive" или "negative")
    predicted_class = "positive" if prediction[0][0] > 0.5 else "negative"

    return jsonify({'label': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
