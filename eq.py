import cv2
import numpy as np
import matplotlib.pyplot as plt

def equlize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # из BGR в LAB
    l, a, b = cv2.split(lab)  # разделение каналов
    equalized_channels = cv2.equalizeHist(l)  # применение к каналу L (яркости)
    lab_eq = cv2.merge((equalized_channels, a, b))  # слияние обработанного канала L с остальными каналами
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result

image = cv2.imread('0.jpg')
eq = equlize(image) # применение эквилизации
clahe_img = cv2.imread('output_3.30_15.19.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
eq_rgb = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
clahe_img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # преобразование в полутоновое
eq_gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
clahe_img_gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)

# вычисление гистограмм
hist_original = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])
hist_clahe_image = cv2.calcHist([clahe_img_gray], [0], None, [256], [0, 256])
# Построение графиков
plt.figure(figsize=(15, 15))

# Оригинальное изображение и его гистограмма
plt.subplot(3, 2, 1)
plt.imshow(image_rgb)
plt.title('Оригинальное изображение', fontname='Times New Roman', fontsize=16)
plt.axis('off')

plt.subplot(3, 2, 2)
plt.plot(hist_original, color='black')
plt.title('Гистограмма оригинального изображения', fontname='Times New Roman', fontsize=16)
plt.xlabel('Интенсивность пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.ylabel('Количество пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.yticks(fontname='Times New Roman', fontsize=16)

# Эквилизированное изображение и его гистограмма
plt.subplot(3, 2, 3)
plt.imshow(eq_rgb)
plt.title('Эквилизация изображения', fontname='Times New Roman', fontsize=16)
plt.axis('off')

plt.subplot(3, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Гистограмма эквилизованного изображения', fontname='Times New Roman', fontsize=16)
plt.xlabel('Интенсивность пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.ylabel('Количество пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.yticks(fontname='Times New Roman', fontsize=16)

plt.subplot(3, 2, 5)
plt.imshow(clahe_img_rgb)
plt.title('Изображение после применения CLAHE', fontname='Times New Roman', fontsize=16)
plt.axis('off')

plt.subplot(3, 2, 6)
plt.plot(hist_clahe_image, color='black')
plt.title('Гистограмма изображения после применения CLAHE', fontname='Times New Roman', fontsize=16)
plt.xlabel('Интенсивность пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.ylabel('Количество пикселей, отн.ед.', fontname='Times New Roman', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.yticks(fontname='Times New Roman', fontsize=16)

plt.subplots_adjust(top=0.9)  # Увеличение отступа сверху
plt.tight_layout()
plt.show()
