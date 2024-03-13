import cv2
import numpy as np
from skimage import io, color, util
import metr_clah as metr


if __name__ == "__main__":
    # Загрузка изображения
    image_path = "1.jpg"
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
        quality_brisque, blur_score = metr.calculate_brisque_quality(blurred)
        quality_brisque = quality_brisque[0] if isinstance(quality_brisque, tuple) else quality_brisque
        # quality_brisque_str = str(quality_brisque)  # Преобразование в строку

        # PSNR
        quality_psnr = metr.calculate_PSNR_quality(original_image, blurred)
        quality_psnr = quality_psnr[0] if isinstance(quality_psnr, tuple) else quality_psnr

        # SSIM
        quality_ssim = metr.calculate_SSIM_quality(original_image, blurred)
        quality_ssim = quality_ssim[0] if isinstance(quality_ssim, tuple) else quality_ssim

        #Градиент
        quality_grad = metr.image_quality_squared_gradient(blurred)
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

    for i in range(len(sigma_list)):
        print(f'Sigma: {sigma_list[i]}, QualityBRISQUE: {quality_brisque_values_cl[i]}, '
              f'QualityPSNR: {quality_psnr_values_cl[i]}, QualitySSIM: {quality_ssim_values_cl[i]}, '
              f'QualityGrad: {quality_grad_values[i]}')
