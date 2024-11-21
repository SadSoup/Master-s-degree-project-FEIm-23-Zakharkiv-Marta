import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # виводитиме лише помилки

from tensorflow.keras.models import load_model
# Отримання шляху до поточного скрипта
script_directory = os.path.dirname(os.path.abspath(__file__))

# завантаження тестових даних
test_images = np.load(os.path.join(script_directory,'test_images.npy'))
test_labels_encoded = np.load(os.path.join(script_directory,'test_labels_encoded.npy'))

# завантаження моделі
if_model_name = input('Якщо бажаєте переглянути результати результати іншої моделі, введіть "y"\n(в іншому випадку буде наведено оцінку моделі, навченої авторкою роботи)\n\nВідповідь: ')
if if_model_name.strip().lower() == 'y':
    model_name = input('Введіть назву моделі, яка знаходиться в одній папці, із цим файлом (без розширення): ')
    model_path = os.path.join(script_directory, model_name + '.h5')
    if model_name and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print('Файл не знайдено або назва порожня. Використовуватиметься модель за замовчуванням.')
        model = load_model(os.path.join(script_directory, 'model.h5'))
else:
    print('Оцінка моделі за замовчуванням:')
    model = load_model(os.path.join(script_directory,'model.h5'))

# ф-я оцінки моделі
def evaluate_model(y_true, model, test_images, class_names=None):

    y = model.predict(test_images, verbose = 0)
    y_pred = np.argmax(y, axis=1)
    # Обчислення основних метрик
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=1))
    
    # Створення матриці помилок
    cm = confusion_matrix(y_true, y_pred)
    
    # Візуалізація матриці помилок
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матриця помилок')
    plt.ylabel('Істинні значення')
    plt.xlabel('Передбачені значення')
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.show()

# Отримуємо унікальні класи
unique_classes = np.unique(test_labels_encoded)
class_names = [f'Class {i}' for i in unique_classes]  # Замініть на реальні назви класів
class_names = ['Басет-гаунд', 'Бігль', 'Англійський сетер', 'Курцхаар','Ньюфаундленд','Сенбернар']

# Оцінюємо модель
evaluation_results = evaluate_model(
    y_true=test_labels_encoded,
    model=model,
    test_images=test_images,
    class_names=class_names
)
