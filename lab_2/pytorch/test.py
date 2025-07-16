import pandas as pd

# Загрузите ваш датасет
df = pd.read_csv('data/emotions.csv')
print(df['label'].unique())
