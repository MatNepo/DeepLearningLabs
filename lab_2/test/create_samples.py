from safetensors import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch.optim as optim

from download_dataset import train_df


# Подготовка данных
class IMDBDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['text'].values
        self.labels = dataframe['label'].apply(lambda x: 1 if x == 'pos' else 0).values  # Конвертация меток

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Создайте векторизатор
vectorizer = CountVectorizer(max_features=5000)  # Ограничьте количество признаков
X_train = vectorizer.fit_transform(train_df['text']).toarray()
y_train = train_df['label'].apply(lambda x: 1 if x == 'pos' else 0).to_numpy()

# Преобразование в PyTorch Dataset
train_dataset = IMDBDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Определение модели LMU (как в предыдущем коде)
# ...

# Настройка обучения
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LMU(embedding_dim=128, hidden_size=64, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
for epoch in range(10):
    model.train()
    for texts, labels in train_loader:
        texts = vectorizer.transform(texts).toarray()  # Преобразование текстов
        inputs = torch.tensor(texts, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Эпоха {epoch + 1}, Потеря: {loss.item()}')

if __name__ == '__main__':
    # Оценка модели на тестовых данных
    model.eval()
    test_dataset = IMDBDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts = vectorizer.transform(texts).toarray()
            inputs = torch.tensor(texts, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Точность на тестовых данных: {accuracy:.4f}')
