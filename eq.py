import cv2
import numpy as np
import matplotlib.pyplot as plt

def equlize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # из BGR в LAB
    l, a, b = cv2.split(lab)  # разделение каналов
    equalized_channels = cv2.equalizeHist(l)  # Применение к каналу L (яркости)
    lab_eq = cv2.merge((equalized_channels, a, b))  # слияние обработанного канала L с остальными каналами
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result

# Загрузка изображения
image = cv2.imread('0.jpg')
eq = equlize(image)
#eq = cv2.imread('output_3.30_15.19.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
eq_rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)

# Вычисление гистограмм оригинального и эквилизированного изображений
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])

# Построение графиков
plt.figure(figsize=(15, 6))

# Оригинальное изображение и его гистограмма
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Оригинальное изображение', fontname='Times New Roman', fontsize=16)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(hist_original, color='black')
plt.title('Гистограмма оригинального изображения', fontname='Times New Roman', fontsize=16)
plt.xlabel('Интенсивность пикселей', fontname='Times New Roman', fontsize=16)
plt.ylabel('Количество пикселей', fontname='Times New Roman', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.yticks(fontname='Times New Roman', fontsize=16)
# Эквилизированное изображение и его гистограмма
plt.subplot(2, 2, 2)
plt.imshow(eq_rgb)
plt.title('Эквилизация изображения', fontname='Times New Roman', fontsize=16)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Гистограмма эквилизованного изображения', fontname='Times New Roman', fontsize=16)
plt.xlabel('Интенсивность пикселей', fontname='Times New Roman', fontsize=16)
plt.ylabel('Количество пикселей', fontname='Times New Roman', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.yticks(fontname='Times New Roman', fontsize=16)
plt.subplots_adjust(top=0.9)  # Увеличение отступа сверху
plt.tight_layout()
plt.show()