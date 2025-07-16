Сейчас на практике чаще используется модель s4 (2021 год), а LMU (2020 год). Для реализации использовал модель LMU, представленную группой Стэнфордских учёных (HIPPO).


Основные ссылки по LMU:

1.1. Статья авторов LMU ([ссылка](https://papers.nips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf))

1.2. Репозиторий keras-lmu, на который ссылаются в статье из пункта [1.1] ([ссылка](https://github.com/nengo/keras-lmu))

1.3. Метод, описанный командой Стэндфордских учёны, по улучшению эффективности LMU ([ссылка](https://arxiv.org/pdf/2008.07669))

1.4. Репозиторий hippo-code, на который ссылаются в статье из пункта [1.3] ([ссылка](https://github.com/HazyResearch/hippo-code/tree/master))

1.5. Пост этой же группы учёных с коротким описанием более эффективного метода ([ссылка](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo))

Тестирование модели проводил с использованием Postman, отправляя на локальный сервер POST запросы с текстом для подачи на вход модели. Вот несколько примеров работы (на 2 скриншоте использовал большое количество сокращений, как и ожидалось, модель не смогла нормально распознать эмоцию, об этом можно судить по выводу в терминал, где показано, что точность предсказания меньше 1%, в остальных случаях предсказания достаточно точные):

[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-49.jpg)
[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-50.jpg)
[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-51.jpg)
[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-52.jpg)
[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-53.jpg)
[Image alt](https://github.com/MatNepo/DeepLearningLabs/new/main/lab_2/images/photo_2024-11-05_11-20-54.jpg)
