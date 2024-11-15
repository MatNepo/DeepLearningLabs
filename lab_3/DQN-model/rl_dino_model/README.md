# Отчёт по реализации игры Dino с использованием обучения с подкреплением (RL)

### 1. Введение

**Задача:** Реализовать решение для игры Dino с использованием подходов 
обучения с подкреплением. Игра представляет собой динамическую среду, в 
которой агент должен избегать препятствий (прыгать или пригибаться), 
получая награды за каждое успешно пройденное препятствие. Задача требует 
от агента реакции на быстро меняющиеся условия, что делает её подходящей 
для применения DQN (Deep Q-Network).

**Мотивация:** Исследование алгоритмов обучения с подкреплением на задаче, 
приближённой к реальной. Прототип позволяет изучить возможности и 
ограничения стандартного DQN и определить направления его улучшения.

---

### 2. Описание модели

**Архитектура:**  
Для решения задачи была использована модель DQN, состоящая из:
- **Свёрточных слоёв** для обработки входных изображений, которые помогают 
выделять важные признаки, такие как препятствия и положение персонажа.
- **Полносвязных слоёв** для оценки Q-значений, что позволяет агенту 
определять оптимальные действия (прыжок, пригибание, или продолжение 
движения).

**Механизмы оптимизации:**  
- **Experience Replay**: Буфер для хранения последовательности прошлых состояний, что позволяет многократно использовать одно и то же состояние для обучения, избегая коррелированных данных.
- **Эпсилон-жадная (eps-greedy) стратегия**: Для выбора действия используется стратегия, балансирующая случайный выбор (исследование) и выбор на основе предсказания модели (эксплуатация).

---

### 3. Этапы реализации

#### Подготовка среды
Среда для обучения агента создана с использованием `gymnasium` и `pygame`. Агент имеет доступ к действиям `JUMP`, `DUCK` и `STAND`, которые позволяют ему избегать препятствий и набирать очки.

#### Реализация модели
Модель `DQN` была настроена так, чтобы обрабатывать входные кадры и 
предсказывать полезность (Q-значения) для каждого действия:
- **Свёрточная часть**: Три слоя с различными ядрами и шагами, 
адаптированные под низкоразмерные изображения, используемые для игры Dino.
- **Полносвязная часть**: Линейные слои, на выходе которых Q-значения для 
каждого действия.

#### Оптимизация и обучение
Процесс обучения состоит из множества эпизодов, в каждом из которых агент 
оценивает состояние, выбирает действие и обновляет модель на основе 
накопленного опыта. В процессе обучения модель:
- Обновляет буфер опыта с фиксированной частотой.
- Регулярно обновляет целевую сеть (`target network`), чтобы снизить 
нестабильность обучения.

---

### 4. Результаты и визуализация

На каждом 50-м эпизоде создаётся GIF-анимация с ходом игры, что 
позволяет наблюдать за изменениями в поведении агента. Эти визуализации 
показывают прогресс, достигнутый агентом: по мере обучения увеличивается 
общая награда, и агент демонстрирует более устойчивое избегание препятствий.

---

---

### 5. Запуск кода

1) В папке [`./type_1/`](https://github.com/MatNepo/DeepLearningLabs/tree/main/lab_3/DQN-model/rl_dino_model/type_1) представлена итоговая реализация проекта, более 
подробная информация по проекту содержится внутри директории в файлах 
`README.md`, `play.ipynb` и `train.ipynb`
2) В папке [`./type_2/`](https://github.com/MatNepo/DeepLearningLabs/tree/main/lab_3/DQN-model/rl_dino_model/type_2) представлен способ реализации, где взаимодействие 
с игрой происходит с помощью эмуляций нажатия клавиш и мыши на реальном 
мониторе

---

### 6. Потенциал для дальнейшего улучшения

Чтобы повысить эффективность и стабильность модели, были предложены 
несколько направлений для дальнейшего развития:
- **Double DQN:** Использование двойного Q-обучения для уменьшения 
вероятности переоценки Q-значений.
- **Prioritized Experience Replay:** Приоритетный выбор значимых состояний 
в буфере опыта для более эффективного обучения на сложных ситуациях.
- **Dueling DQN:** Разделение оценок состояния и действий для более 
детальной оценки полезности каждого действия.
