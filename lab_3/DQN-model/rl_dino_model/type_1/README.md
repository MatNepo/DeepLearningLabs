# Использование кода в проекте:

---
## 1. Установка библиотек

Перед тем как начать работу, необходимо подготовить окружение, для автоматического сопоставления версий пакетов:

### 1.1. Установка пакетов для `./type_2/`

```bash
conda create --name env_pt python=3.11
conda activate env_pt

```

Далее установим PyTorch с поддержкой CUDA, в данном случае будет использоваться версия `pytorch-cuda=11.8`. После установке выполним проверку того, что все пакеты были успешно установлены:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda list torch
```
*здесь автоматически установится необходимая версия NumPy

В качестве алгоритма обучения с подкреплением, будет использоваться Stable Baselines3 (SB3):

```bash
conda install conda-forge::stable-baselines3
```

Pytesseract — это обертка для инструмента OCR (распознавания текста) под названием Tesseract, разработанного Google. Эта библиотека позволяет извлекать текст из изображений.

```bash
conda install conda-forge::pytesseract
```

Чтобы воспроизвести функциональность ввода с помощью мыши и клавиатуры, установим библиотеку PyDirectInput, которая является более быстрым аналогом PyAutoGUI:

```bash
conda install pydirectinput
```

Далее установим MSS (Multiple Screen Shots) — это библиотека для захвата скриншотов. Будем использовать именно её, так как она легко интегрируется с opencv:

```bash
conda install python-mss
```

Установим остальные библиотеки:
```bash
conda install opencv matplotlib gymnasium
```

### 1.2. Установка пакетов для `./type_1/`

```bash
conda install argparse envs pygame  # для play.py
conda install PIL collections datetime itertools os shutil random  # для train.py
conda install tqdm  # для experiment.py
```


---
## 2. Запуск обучения модели

```bash
python train.py
```
При желании в файле `train.py` параметры класса Trainer для обучения агента 
можно задать по-другому, более подробно можно прочесть в `train.ipynb`

---
## 3. Запуск для игры вручную с клавиатуры

```bash
python play.py human
```

---
## 4. Запуск для игры с агентом

```bash
python play.py ai -m model.pth
```
Используются параметры из model.pth

---
## *5. Запуск игры для проведения экспериментов с обученным агентом

```bash
python experimental.py
 ```
Подробнее об использовании см. в файле (нужен в основном для представления 
информации в виде графика, основной код состоит только из `train.py` и 
`plsy.py`).