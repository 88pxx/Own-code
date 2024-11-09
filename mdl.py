import tensorflow as tf

from keras import layers, models
import matplotlib.pyplot as plt
from keras.src.callbacks import ModelCheckpoint
from keras.src.callbacks import TensorBoard




learn_dir_path = r'D:\\PYTHON\\PYTH_PROJECTS\\Brains\\Data\\ALL'

one_step_size = 20  # число изображений через которое обновляются параметры
img_height = 300
img_width = 300

# ___ПРЕДОБРАБОТКА___

learn_ds = tf.keras.preprocessing.image_dataset_from_directory(
    learn_dir_path,
    validation_split=0.25,  # 25 % валидация, 75% обучение
    subset="training",  # выбранные 75% для обучения
    seed=111,
    image_size=(img_height, img_width),
    batch_size=one_step_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    learn_dir_path,
    validation_split=0.25,
    subset="validation",  # выбранные 25% для тестирования
    seed=111,
    image_size=(img_height, img_width),
    batch_size=one_step_size)

# ___МОДЕЛЬ___

n_class = 4

model = models.Sequential([
    # слой нормализации
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),  # 3 число rgb каналов

    layers.Conv2D(32, (3, 3), activation='relu'),  # сверточный слой извлечения признака
    layers.MaxPooling2D((2, 2)),  # слой пулинга понижение размерности

    layers.Conv2D(64, (3, 3), activation='relu'),  # 32, 64, 128 нейронов в слое
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Полносвязный классификацинный слой + преобразование в одномерный тип данных
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_class)
])

# ___СБОРКА___


model.compile(optimizer='adam',  # функция потерь адаптивная оценка моментов
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  # оценка по точности


#___СОХРАНЕНИЕ МОДЕЛИ___

checkpoint_callback =tf.keras.callbacks.ModelCheckpoint( r'Main/1_model.keras',
    #filepath = '1_model_checkpoint.h5',
    save_best_only = True,
    monitor = 'val_loss', # метрика отслеживания
    mode = 'min', # минимизация или максимизация метрики
    save_weights_only = False,
    verbose = 1 #  условия сохранения
)






#___ТЕЛЕМЕТРИЯ___
telemetry = TensorBoard (
    log_dir ='data',
    histogram_freq = 1,
    write_graph= True,
    write_images = True
)





# ___ОБУЧЕНИЕ___

epochs = 5
history = model.fit(
    learn_ds,
    validation_data = test_ds,
    epochs = epochs,   # число проходов по выборке
    callbacks = [checkpoint_callback, telemetry]
)

# ___ОЦЕНКА ТОЧНОСТИ___

# Точности
acc = history.history['accuracy']  # обучающая
test_acc = history.history['val_accuracy']  # тестовая

# Потери
loss = history.history['loss']  # обучающая
test_loss = history.history['val_loss']  # тестовая

epochs_range = range(epochs)

# __ГРАФИКИ__

# График точности
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# График потерь
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
