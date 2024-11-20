import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import multiprocessing
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # виводитиме лише помилки

# виділення ядер для ефективнішої роботи
num_workers = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() > 3 else 1

try:
    # задання користувачем кількості епох
    epoch_number = int(input('Введіть кількість епох\n(для високої точності рекомендовано 300 епох, однак модель вчитиметься орієнтовно 13хв; для швидкої перевірки можна задати до 10 епох)\n\nЕпохи: '))
except ValueError:
    print("Наступного разу введіть число.")
    exit()

# Отримання шляху до поточного скрипта
script_directory = os.path.dirname(os.path.abspath(__file__))

# завантажуємо дані
train_images = np.load(os.path.join(script_directory,'train_images.npy'))
val_images = np.load(os.path.join(script_directory,'val_images.npy'))

val_labels_encoded = np.load(os.path.join(script_directory,'val_labels_encoded.npy'))

train_labels_categorical = np.load(os.path.join(script_directory,'train_labels_categorical.npy'))
val_labels_categorical = np.load(os.path.join(script_directory,'val_labels_categorical.npy'))

# задаємо параметри аугментації
train_datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator()

# задаємо розмір батчу
train_generator = train_datagen.flow(train_images, train_labels_categorical, batch_size=20)
validation_generator = validation_datagen.flow(val_images, val_labels_categorical, batch_size=20)


# створення графіку навчання
def plot_history(history,model,val_images,val_encoded):

    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]


    epochs   = range(1,len(acc)+1)

    stats = epoch_stats_callback.get_epoch_stats(len(acc))
    time_spent = round(stats[0],3)
    seconds = math.floor(time_spent)
    ms = int(round((time_spent - seconds),3)*1000)
    time_str = f"{seconds}с, {ms}мс"
    if time_spent>=60 and time_spent<3600:
        mins = int(math.floor(time_spent/60))
        time_str = f"{mins}хв, {seconds-mins*60}с, {ms}мс"
    elif time_spent>=3600:
        hours = int(math.floor(seconds/3600))
        mins = int(math.floor(seconds/60)-60*hours)
        time_str = f"{hours}год, {mins}хв, {seconds-hours*3600-mins*60}c, {ms}мс"
    
    # Отримуємо предсказання на тестовій вибірці
    y_pred = model.predict(val_images, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    report = classification_report(val_encoded, y_pred_classes, zero_division=1)
    print('\n',report)

    # Обчислимо середню точність
    val_acc_compare = history.history['val_accuracy'][-10:]
    mean_val_acc = np.mean(val_acc_compare)

    print(f'Середня валідаційна точність останніх 10 епох: {mean_val_acc:.2f}')

    plt.plot  ( epochs,     acc , label = 'Точність для тренувального набору', color='black', linestyle='-')
    plt.plot  ( epochs, val_acc , label = 'Точність для валідаційного набору', color='red', linestyle='--')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.title (f'Точність тренування і валідації ({time_str})')
    plt.legend()
    plt.show()

# "ф-я" зворотнього виклику 
class EpochStatsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EpochStatsCallback, self).__init__()
        self.epoch_stats = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        # Додаємо час та точність для цієї епохи до словника
        self.epoch_stats[epoch + 1] = (epoch_time, accuracy, val_accuracy)

    def get_epoch_stats(self, epoch):
        """Повертає статистику для конкретної епохи та сумарний час."""
        if epoch not in self.epoch_stats:
            return "Епохи не існує"
        
        total_time = 0
        for i in range(1, epoch + 1):
            total_time += self.epoch_stats[i][0]  # Додаємо час кожної епохи

        accuracy = self.epoch_stats[epoch][1]
        val_accuracy = self.epoch_stats[epoch][2]

        return total_time, accuracy, val_accuracy
    
epoch_stats_callback = EpochStatsCallback()

# конфігуруємо модель
def model_2():
    model = tf.keras.models.Sequential([
        # шари згортки та підвибірки
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(256, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(256, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4), # щар відсіву

        # повнозв'язні шари
        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Dense(64,  activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        
        # вихідний шар
        tf.keras.layers.Dense(6, activation='softmax') 
    ])
    return model

# ініціалізація моделі
model=model_2()

# компіляція моделі
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(
    train_generator,
    epochs=epoch_number,
    validation_data=validation_generator,
    verbose=1,
    workers=num_workers, 
    use_multiprocessing=False,  
    callbacks=[epoch_stats_callback]
)

plot_history(history, model, val_images, val_labels_encoded)

if_save = input('Чи бажаєте Ви зберегти модель[y/n]: ')
if if_save.strip().lower() == 'y':
    model_name = input('Назвіть модель (без розширення): ')
    model.save(os.path.join(script_directory,model_name + '.h5'))
    print('Модель збережено.')
else:
    print('Модель не збережено')