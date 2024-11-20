import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import os

# ф-я для перетворення зображень у масиви numpy
def load_images_from_folder(folder, size=(150, 150)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = load_img(img_path, target_size=size)
            img_array = img_to_array(img) / 255.0  # Нормалізуємо зображення
            images.append(img_array)
            labels.append(label)  # Додаємо мітку
    unique_labels = np.unique(np.array(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)} 
    labels_encoded = np.array([label_to_index[label] for label in np.array(labels)])    
    labels_categorical = to_categorical(labels_encoded)   

    return np.array(images), labels_encoded, labels_categorical

# Отримання шляху до поточного скрипта
script_directory = os.path.dirname(os.path.abspath(__file__))

# Перетворення зображень у масиви numpy
train_images, _, train_labels_categorical = load_images_from_folder(os.path.join(script_directory,'images/train'))
val_images, val_labels_encoded, val_labels_categorical = load_images_from_folder(os.path.join(script_directory,'images/validation'))
test_images, test_labels_encoded, _ = load_images_from_folder(os.path.join(script_directory,'images/test'))

# Збереження у форматі .npy в папці файлу
np.save(os.path.join(script_directory,'train_images.npy'), train_images)
np.save(os.path.join(script_directory,'train_labels_categorical.npy'), train_labels_categorical)

np.save(os.path.join(script_directory,'val_images.npy'), val_images)
np.save(os.path.join(script_directory,'val_labels_encoded.npy'), val_labels_encoded)
np.save(os.path.join(script_directory,'val_labels_categorical.npy'), val_labels_categorical)

np.save(os.path.join(script_directory,'test_images.npy'), test_images)
np.save(os.path.join(script_directory,'test_labels_encoded.npy'), test_labels_encoded)
