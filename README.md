# vi_home_task_4
Домашняя работа по распознованию ключевых точек на лице

run_test.py - тестранер. Отличается от исходного закомментированным fast_train 
https://github.com/re-gor/vi_home_task_4/blob/master/run_tests.py#L118

`best_grey.hdf5` - лучшая модель для чернобелых картинок. Картинки на вход нейросети подаются чернобелыми. 
В остальном совпадает с `facepoints_model.hdf5`

`facepoints_model.hdf5` - модель-ответ. 

`detection.py` - файл-ответ. Помимо функции `train_detector` и `detect` cодержит генератор и функции для преобразования картинок

#### ещё

`actual_learning.ipynb` - обучал и подбирал архитектуру по факту тут
`crossval.hdf5` - модель, обученнная на кросвалидационной выборке
