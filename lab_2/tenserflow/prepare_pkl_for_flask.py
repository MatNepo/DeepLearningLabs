import pickle

from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# Загрузка данных IMDb с Hugging Face
dataset = load_dataset("jahjinx/IMDb_movie_reviews")

# Преобразуем в pandas DataFrame
df_train = pd.DataFrame(dataset['train'])

# Извлечение текстов
texts = df_train['text'].values

# Создание и обучение токенизатора
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Сохранение токенизатора в файл
with open('./results/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved to ./results/tokenizer.pkl")
