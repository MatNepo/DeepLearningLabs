# model.py

import tensorflow as tf
from keras_lmu import LMUCell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RNN, Dense
import pickle

# Загрузка предобработанных данных
with open('processed_data.pkl', 'rb') as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

# Настройка гиперпараметров
memory_d = 128  # Размер памяти LMU
order = 8       # Порядок функции Лежандра
units = 64      # Количество нейронов
epochs = 10     # Количество эпох
batch_size = 64 # Размер батча

# Создание LMU-ячейки
lmu_cell = LMUCell(memory_d=memory_d, order=order, units=units)
lmu_layer = RNN(lmu_cell)

# Построение модели
model = Sequential([
    lmu_layer,
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Бинарная классификация
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Колбэки для мониторинга и остановки
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Обучение модели
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1  # Вывод прогресса
)

# Сохранение модели
model.save('lmu_model.h5')
