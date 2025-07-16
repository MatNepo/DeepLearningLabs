import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.utils.rnn import pad_sequence

class EmotionsDataset(Dataset):
    def __init__(self, csv_file):
        # Указываем разделитель ';' при загрузке CSV
        self.data = pd.read_csv(csv_file, sep=';')
        print("Columns in dataset:", self.data.columns.tolist())  # Выводим названия столбцов

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        inputs = item['text']  # Получаем текст
        label = item['label']  # Получаем метку
        return inputs, label

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Пример простой агрегации
        x = self.fc(x)
        return x

def train_model(config):
    # Загружаем данные
    dataset = EmotionsDataset(config['csv_file'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Параметры модели
    vocab_size = 10000  # Замените на размер вашего словаря
    embed_dim = 128
    num_classes = 5  # Замените на количество классов в вашем датасете

    # Создаем модель, критерий и оптимизатор
    model = SimpleModel(vocab_size, embed_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    model.train()
    for epoch in range(config['num_epochs']):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Здесь необходимо преобразовать inputs в числовой формат
            inputs = torch.tensor([text_to_index(text) for text in inputs])  # Замените на ваш метод
            labels = torch.tensor(labels.tolist())

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

def text_to_index(text):
    # Пример функции, которая преобразует текст в числовые индексы
    return [ord(char) for char in text]  # Замените на свой токенизатор

def collate_fn(batch):
    inputs, labels = zip(*batch)  # Разделяем входные данные и метки
    inputs = [torch.tensor(text_to_index(text)) for text in inputs]  # Преобразуем текст в индексы
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Паддинг до одинаковой длины
    labels = torch.tensor(labels)
    return inputs, labels

def train_model(config):
    # Загружаем данные
    dataset = EmotionsDataset(config['csv_file'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)  # Указываем collate_fn

    # Параметры модели
    vocab_size = 10000  # Замените на размер вашего словаря
    embed_dim = 128
    num_classes = 5  # Замените на количество классов в вашем датасете

    # Создаем модель, критерий и оптимизатор
    model = SimpleModel(vocab_size, embed_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    model.train()
    for epoch in range(config['num_epochs']):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_model(config)
