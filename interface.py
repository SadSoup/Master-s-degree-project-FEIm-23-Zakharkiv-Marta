import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # виводитиме лише помилки

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Отримання шляху до поточного скрипта
script_directory = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(script_directory,'model.h5'))
class_indicies = {0: 'басет-гаунд',
 1: 'бігль',
 2: 'англійський сетер',
 3: 'курцхаар',
 4: 'ньюфаундленд',
 5: 'сенбернар'}

# конвертація зображення в необхідний для моделі формат
def load_image_from_folder(img_path, size=(150, 150)):

    img = load_img(img_path, target_size=size)
    img_array = img_to_array(img) / 255.0  # Нормалізуємо зображення
    
    return img_array

# налаштування полотна
def full_image(event=None):
    global resized_imagetk

    if event:
        canvas_width = event.width
        canvas_height = event.height
    else:
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

    canvas_ratio = canvas_width / canvas_height

    if canvas_ratio > image_ratio:
        height = int(canvas_height)
        width = int(height * image_ratio)
    else:
        width = int(canvas_width)
        height = int(width / image_ratio)

    resized_image = image_to_classify.resize((width, height))
    resized_imagetk = ImageTk.PhotoImage(resized_image)
    canvas.create_image(int(canvas_width / 2), int(canvas_height / 2), anchor='center', image=resized_imagetk)


# вибір зображення
def choose_image():
    global image_to_classify, image_ratio, image_tk, file_path
    try:
        file_path_try = filedialog.askopenfilename(initialdir = script_directory, title='Select a file', filetypes=(('jpg files', '*.jpg'), ('all files', '*.*')))
        
        if not file_path_try:
            return  # Вихід з функції, якщо файл не обрано
        
        file_path = file_path_try
        image_to_classify = Image.open(file_path)
        image_ratio = image_to_classify.size[0]/ image_to_classify.size[1]
        image_tk = ImageTk.PhotoImage(image_to_classify)
        full_image(None)
        canvas.bind('<Configure>', full_image)
        poss_label.config(text=f'Порода:')
        breds_label.config(text=f'басет-гаунд:\nбігль:\nанглійський сетер:\nкурцхаар:\nньюфаундленд:\nсенбернар: \n\n\n\n\n\n')


    except (IOError,PermissionError):
        # Обробка помилок відкриття файлу
        messagebox.showerror("Помилка", "Не вдалося відкрити зображення. Оберіть інший файл.")

# розпізнавання зображення
def classify_image():
    global model

    # # Перевірка, чи є зображення
    if 'file_path' not in globals() or file_path is None:
        messagebox.showwarning("Попередження", "Спочатку оберіть зображення")
        return
    if not file_path:
            return  # Вихід з функції, якщо файл не обрано
    # Передбачення
    img_array = load_image_from_folder(file_path)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)

    poss_label.config(text=f'Порода:\nЗ імовірністю \n{(prediction[0][predicted_class]):.2f} це \n{class_indicies[predicted_class]}')
    breds_label.config(text=f'басет-гаунд:{(prediction[0][0]):.2f}\nбігль:{(prediction[0][1]):.2f}\nанглійський сетер:{(prediction[0][2]):.2f}\nкурцхаар:{(prediction[0][3]):.2f}\nньюфаундленд:{(prediction[0][4]):.2f}\nсенбернар:{(prediction[0][5]):.2f} \n\n\n\n\n\n')

def choose_model():
    global model
    try:
        model_path = filedialog.askopenfilename(initialdir = script_directory, title='Select a file', filetypes=(('h5 files', '*.h5'), ('all files', '*.*')))
        if not model_path:
            return  # Вихід з функції, якщо файл не обрано
        model = tf.keras.models.load_model(model_path)
        # Збереження назви моделі
        model_name_str = os.path.basename(model_path) 

        model_name.config(text=f'{model_name_str}\n\n')
        messagebox.showinfo("Успіх", f"Модель {model_name_str} успішно завантажена")
    except(ValueError, OSError) as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити модель: {e}")

    

# setup
window = tk.Tk()
window.geometry('900x600')
window.title('Image classifier')

# сітка
window.columnconfigure((0,1,2,3), weight=1, uniform = 'a')
window.rowconfigure(0, weight=1)

# canvas 
canvas = tk.Canvas(window,  bd=0,highlightthickness=0, relief='ridge')
canvas.grid(column=1,columnspan=3,row=0, rowspan=2, sticky='nsew')

# кнопки
buttonFrame = tk.Frame(window)

button = ttk.Button(buttonFrame, text='Обрати зображення', command=choose_image)
button.pack(anchor='w') #відстань між об'єктами
button_to_classify = ttk.Button(buttonFrame, text='Яка це порода?', command=classify_image)
button_to_classify.pack(anchor='w') #відстань між об'єктами
buttonFrame.grid(column=0,row=0, sticky='nsew') # розміщення кнопки вгорі в куті - sticky


# мітки
poss_label = ttk.Label(buttonFrame, text= 'Порода:', font = ('Helvetica', 14, 'bold'),anchor='w', justify='left')
poss_label.pack(pady=20,fill='x')


breedsFrame = ttk.Frame(window)
all_poss_label = ttk.Label(breedsFrame, text= 'Імовірності приналежності \nдо породи:\n', font = ('Helvetica', 11, 'bold'),anchor='w', justify='left')
all_poss_label.pack(anchor='w')
breds_label = ttk.Label(breedsFrame, text= 'басет-гаунд:\nбігль:\nанглійський сетер:\nкурцхаар:\nньюфаундленд:\nсенбернар: \n\n\n\n\n\n', font = ('Helvetica', 11),anchor='w', justify='left')
breds_label.pack(anchor='w')
model_label = ttk.Label(breedsFrame, text='Використовується модель:',font = ('Helvetica', 11, 'bold'),anchor='w', justify='left')
model_label.pack(anchor='w')
model_name = ttk.Label(breedsFrame, text='За замовчуванням\n\n')
model_name.pack(anchor='w')
button_model = ttk.Button(breedsFrame, text='Обрати іншу модель',  command=choose_model)
button_model.pack(anchor='sw')
breedsFrame.grid(column=0,row=1, sticky='nsew') # розміщення кнопки вгорі в куті - sticky


# run
window.mainloop()