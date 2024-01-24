import cv2

# Открытие изображения в оттенках серого
image_path = '9.jpg'
image = cv2.imread(image_path, 0)

# Применение эквализации гистограммы
equalized_image = cv2.equalizeHist(image)

# Отображение исходного и эквализованного изображения
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()