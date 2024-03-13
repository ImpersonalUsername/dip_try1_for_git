import cv2
import numpy as np
from skimage import io, color, util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import metr_clah as metr

def apply_clahe(image, clip_limit):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(5, 5))
    clahe_image = clahe.apply(gray_image)
    result_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

    return result_image
def image_quality_squared_gradient(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    squared_gradient = gradient_x**2 + gradient_y**2
    quality = np.mean(squared_gradient)
    return quality

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = gauss.astype(np.uint8)
    noisy = cv2.add(image, gauss)
    return noisy

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

#Jpeg to BMP
def convert_bmp_to_jpeg_opencv(img, output_path, compression_quality):
    try:
        # Проверка на успешное чтение изображения
        if img is not None:
            # Сохранение изображения в формате JPEG с заданной степенью сжатия
            #cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
            #print(f"Изображение сохранено в {output_path} с качеством {compression_quality}%")

# imencode() преобразует (кодирует) форматы изображений в потоковые данные и сохраняет их в кеше памяти.
            _, compressed_image_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
            # перевод в массив NumPy
            compressed_image = cv2.imdecode(compressed_image_encoded, 1)
            # # Оценка качества
            # brisque_score, blur_score = calculate_brisque_quality(compressed_image)
            # psnr_value = calculate_PSNR_quality(img, compressed_image)
            # ssim_value = calculate_SSIM_quality(img, compressed_image)
            #
            # print(f"Степень сжатия: {compression_quality}%")
            # print(f"BRISQUE Score: {brisque_score[0]}")
            # print(f"Blur Score: {blur_score}")
            # print(f"PSNR: {psnr_value}")
            # print(f"SSIM: {ssim_value}")
            return compressed_image

        else:
            print("Не удалось прочитать изображение.")

    except Exception as e:
        print(f"Ошибка при преобразовании: {e}")
    return
def add_salt_pepper_noise(image, amount):
    s_vs_p = 0.5
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


if __name__ == "__main__":
    # Загрузка изображения
    image_path = "F://PycharmProjects//blur_salt-papper-jpg//1.jpg"
    img = io.imread(image_path)
    original_image = io.imread(image_path)

    # Указание пути для сохранения JPEG-изображения
    jpeg_path = "F://PycharmProjects//blur_salt-papper-jpg//I06"
    compressed_image = []
    qualities = []
    quality_brisque_values_cl = []
    output_images = []
    quality_psnr_values_cl = []
    quality_ssim_values_cl = []
    quality_grad_values = []
    amount_list = []
    #for quality in np.arange(100, -5, -5):
    for clip_limit in np.arange(0.1, 100, 0.1):
        #output_image_path = f'{jpeg_path}_{quality}.jpg'
        img_jpeg = apply_clahe(img, clip_limit)
        compressed_image.append(img_jpeg)
        #BRISQUE
        quality_brisque, blur_score = calculate_brisque_quality(img_jpeg)
        quality_brisque = quality_brisque[0] if isinstance(quality_brisque, tuple) else quality_brisque

        # PSNR
        quality_psnr = calculate_PSNR_quality(img, img_jpeg)
        quality_psnr = quality_psnr[0] if isinstance(quality_psnr, tuple) else quality_psnr

        # SSIM
        quality_ssim = calculate_SSIM_quality(img, img_jpeg)
        quality_ssim = quality_ssim[0] if isinstance(quality_ssim, tuple) else quality_ssim

        #Градиент
        quality_grad = image_quality_squared_gradient(img_jpeg)
        quality_grad = quality_grad[0] if isinstance(quality_grad, tuple) else quality_grad

        # Создаем BGR-изображение для добавления текста
        height, width, channels = img_jpeg.shape  # получаем высоту, ширину и количество каналов изображения
        clahe_result_bgr = img_jpeg  # Берем исходное трехканальное изображение
        text = f'clipLimit: {clip_limit:.2f}\nQualityGrad {quality_grad}\nQualityBRISQUE: {quality_brisque:.2f}\nQualityPSNR: {quality_psnr:.2f}\nQualitySSIM: {quality_ssim:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Коэффициент для масштабирования шрифта
        font_color = (100, 255, 255)
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = (height + text_size[1]) // 2

        # Разбиваем текст на строки и пишем каждую строку отдельно
        line_spacing = 5
        lines = text.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(clahe_result_bgr, line, (text_x, text_y + i * (text_size[1] + line_spacing)), font, font_scale,
                        font_color, font_thickness)
        qualities.append(clip_limit)
        # Сохранение параметров
        quality_brisque_values_cl.append(quality_brisque)
        quality_psnr_values_cl.append(quality_psnr)
        quality_ssim_values_cl.append(quality_ssim)
        quality_grad_values.append(quality_grad)
        # Генерируем имя файла для сохранения
        output_image_path = f'output_{clip_limit:.2f}_{quality_brisque:.2f}_{quality_psnr:.2f}_{quality_ssim:.2f}.jpg'
        cv2.imwrite(output_image_path, cv2.cvtColor(clahe_result_bgr, cv2.COLOR_BGR2RGB))

        output_images.append((clahe_result_bgr, quality_brisque))
    for i in range(len(qualities)):
        print(f'clipLimit: {qualities[i]}, QualityBRISQUE: {quality_brisque_values_cl[i]}, '
              f'QualityPSNR: {quality_psnr_values_cl[i]}, QualitySSIM: {quality_ssim_values_cl[i]}, '
              f'QualityGrad: {quality_grad_values[i]}')