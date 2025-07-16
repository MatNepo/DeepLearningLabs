import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from lmu_model import LMUModel  # Импортируйте вашу модель
from preprocess import load_data, preprocess_data
import pandas as pd
import os

# Установка устройства для использования GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры
input_size = 300  # Укажите размерность входных данных
hidden_size = 128
output_size = 6  # Количество эмоций
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Создание папки для сохранения модели и данных
output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)

# Файл для сохранения статистики обучения
stats_file_path = os.path.join(output_dir, "training_stats.csv")

if __name__ == "__main__":
    print("Используемое устройство:", device)

    # Загрузка и обработка данных
    df = load_data(r'D:\Users\Legion\datasets\emotions\text.csv')  # Укажите путь к вашему файлу
    X_train_padded, X_test_padded, y_train_tensor, y_test_tensor, label_encoder, vocab = preprocess_data(df)

    # Создание DataLoader
    train_dataset = TensorDataset(X_train_padded, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    vocab_size = len(vocab)  # Получите размер словаря
    model = LMUModel(input_size, hidden_size, output_size, vocab_size).to(device)

    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Списки для хранения статистики
    train_losses = []
    train_accuracies = []

    # Создание объекта SummaryWriter
    writer = SummaryWriter('runs/experiment_name')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Логирование каждые N шагов
            if batch_idx % 10 == 0:  # Каждые 10 шагов
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy/train', 100 * correct / total, epoch * len(train_loader) + batch_idx)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/epoch', accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    writer.close()

    # Сохранение модели
    model_file_path = os.path.join(output_dir, 'lmu_model.pth')
    torch.save(model.state_dict(), model_file_path)

    # Сохранение статистики в CSV
    stats_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Loss': train_losses,
        'Accuracy': train_accuracies
    })
    stats_df.to_csv(stats_file_path, index=False)

    print(f"Model saved to: {model_file_path}")
    print(f"Training stats saved to: {stats_file_path}")
