import cv2
import numpy as np
from skimage import io, color, util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def image_quality_squared_gradient(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    squared_gradient = gradient_x**2 + gradient_y**2
    quality = np.mean(squared_gradient)
    return quality

def calculate_brisque_quality(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    brisque_score = cv2.quality.QualityBRISQUE_compute(image, "brisque_model_live.yml", "brisque_range_live.yml")
    return brisque_score, blur_score
# PSNR
def calculate_PSNR_quality(image1, image2):
    if len(image1.shape) == 3:
        image1 = color.rgb2gray(image1)
    if len(image2.shape) == 3:
        image2 = color.rgb2gray(image2)
    mse = ((image1 - image2) ** 2).mean()
    psnr_value = 10 * (np.log10((255 ** 2) / mse))
    return psnr_value

#SSIM
def calculate_SSIM_quality(image1, image2):
    if len(image1.shape) == 3:
        image1 = color.rgb2gray(image1)
    if len(image2.shape) == 3:
        image2 = color.rgb2gray(image2)
    data_range = 1.0 if image1.dtype == np.float64 else 255
    ssim_value = ssim(image1, image2, data_range=data_range)
    return ssim_value

if __name__ == "__main__":
    # Загрузка изображения
    image_path = "I06.jpg"
    original_image = io.imread(image_path)
    blurred_images = []
    # Начальное значение сигма, инкремент и количество итераций
    initial_sigma = 1.0
    sigma_increment = 1.0
    num_iterations = 50
    quality_brisque_values_cl = []
    output_images = []
    quality_psnr_values_cl = []
    quality_ssim_values_cl = []
    quality_grad_values = []
    # Применение размытия фильтром Гаусса с изменяющимися параметрами сигма
    sigma = initial_sigma
    sigma_list = []

    for i in range(num_iterations):
        blurred = cv2.GaussianBlur(original_image, (0, 0), sigma)
        blurred_images.append(blurred)
        # BRISQUE
        quality_brisque, blur_score = calculate_brisque_quality(blurred)
        quality_brisque = quality_brisque[0] if isinstance(quality_brisque, tuple) else quality_brisque
        # quality_brisque_str = str(quality_brisque)  # Преобразование в строку

        # PSNR
        quality_psnr = calculate_PSNR_quality(original_image, blurred)
        quality_psnr = quality_psnr[0] if isinstance(quality_psnr, tuple) else quality_psnr

        # SSIM
        quality_ssim = calculate_SSIM_quality(original_image, blurred)
        quality_ssim = quality_ssim[0] if isinstance(quality_ssim, tuple) else quality_ssim

        #Градиент
        quality_grad = image_quality_squared_gradient(blurred)
        quality_grad = quality_grad[0] if isinstance(quality_grad, tuple) else quality_grad

        # Создаем BGR-изображение для добавления текста
        height, width, channels = blurred.shape  # получаем высоту, ширину и количество каналов изображения
        clahe_result_bgr = blurred  # Берем исходное трехканальное изображение
        text = f'Sigma: {sigma}\nQualityGrad {quality_grad}\nQualityBRISQUE: {quality_brisque:.2f}\nQualityPSNR: {quality_psnr:.2f}\nQualitySSIM: {quality_ssim:.2f}\n'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Коэффициент для масштабирования шрифта
        font_color = (100, 255, 255)  # Зеленый цвет (BGR формат)
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = (height + text_size[1]) // 2
        # text_y = 10 + text_size[0]  # Верхний левый угол

        # Разбиваем текст на строки и пишем каждую строку отдельно
        line_spacing = 5
        lines = text.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(clahe_result_bgr, line, (text_x, text_y + i * (text_size[1] + line_spacing)), font, font_scale,
                        font_color, font_thickness)
        sigma_list.append(sigma)
        sigma += sigma_increment
        # Сохранение параметров
        quality_brisque_values_cl.append(quality_brisque)
        quality_psnr_values_cl.append(quality_psnr)
        quality_ssim_values_cl.append(quality_ssim)
        quality_grad_values.append(quality_grad)

        # Генерируем имя файла для сохранения
        output_image_path = f'output_{quality_brisque:.2f}_{quality_psnr:.2f}_{quality_ssim:.2f}.jpg'
        cv2.imwrite(output_image_path, cv2.cvtColor(clahe_result_bgr, cv2.COLOR_BGR2RGB))

        output_images.append((clahe_result_bgr, quality_brisque))


    # Вывод изображений
    # cv2.imshow("Original Image", original_image)

    # for i in range(num_iterations):
    #     cv2.imshow(f"Blurred Image (Sigma={initial_sigma + i * sigma_increment})", blurred_images[i])
    for i in range(len(sigma_list)):
        print(f'Sigma: {sigma_list[i]}, QualityBRISQUE: {quality_brisque_values_cl[i]}, '
              f'QualityPSNR: {quality_psnr_values_cl[i]}, QualitySSIM: {quality_ssim_values_cl[i]}, '
              f'QualityGrad: {quality_grad_values[i]}')

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
