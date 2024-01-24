import cv2
import numpy as np

def image_quality_squared_gradient(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    squared_gradient = gradient_x**2 + gradient_y**2
    quality = np.mean(squared_gradient)
    return quality

image = cv2.imread('output_77.98_69.93_0.60.jpg')


# Оценка качества изображения по квадрату градиента
quality = image_quality_squared_gradient(image)

print('Качество изображения (Squared Gradient):', quality)