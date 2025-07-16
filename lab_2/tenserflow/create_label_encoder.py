import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Путь к датасету
dataset_folder = './data/'
dataset_name = 'emotions.csv'

# Загрузка данных
df_train = pd.read_csv(dataset_folder + dataset_name)
y_train = df_train['label'].values

# Создание и обучение LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Сохранение обученного LabelEncoder
with open('./results/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("LabelEncoder успешно сохранён.")
