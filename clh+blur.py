import cv2
import numpy as np
from skimage import io, color, util
from skimage.metrics import structural_similarity as ssim

def calculate_tile_grid_size(image):
    min_pixels_per_block = 20000
    max_pixels_per_block = 30000

    height, width = image.shape[:2]

    total_pixels = width * height
    best_tile_grid_size = (1, 1)
    best_num_blocks = 1

    # Попробуем разные комбинации размеров блоков, начиная с 1x1 и увеличивая до половины размера изображения
    for block_size in range(1, min(width, height) + 1):
        num_horizontal_blocks = width // block_size
        num_vertical_blocks = height // block_size
        total_blocks = num_horizontal_blocks * num_vertical_blocks

        if min_pixels_per_block <= block_size ** 2 <= max_pixels_per_block and total_blocks <= total_pixels:
            if total_blocks > best_num_blocks:
                best_tile_grid_size = (num_horizontal_blocks, num_vertical_blocks)
                best_num_blocks = total_blocks

    return best_tile_grid_size
def apply_clahe_1(image, clipLimit, tileGridSize):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) #  из BGR в LAB
    l, a, b = cv2.split(lab) # разделение каналов
    clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize) # Применение CLAHE к каналу L (яркости)
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b)) # слияние обработанного канала L с остальными каналами
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result
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
# imencode() преобразует (кодирует) форматы изображений в потоковые данные и сохраняет их в кеше памяти.
            _, compressed_image_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
            # перевод в массив NumPy
            compressed_image = cv2.imdecode(compressed_image_encoded, 1)
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
    #image_path = "F://PycharmProjects//blur_salt-papper-jpg//0.jpg"
    img = cv2.imread('F://PycharmProjects//blur_salt-papper-jpg//0.jpg')
    #original_image = cv2.imread(image_path)

    # Указание пути для сохранения JPEG-изображения
    compressed_image = []
    qualities = []
    quality_brisque_values_cl = []
    output_images = []
    quality_psnr_values_cl = []
    quality_ssim_values_cl = []
    quality_grad_values = []
    amount_list = []
    tile_generation = calculate_tile_grid_size(img)
    print(tile_generation)
    #for quality in np.arange(100, -5, -5): # относится к сжатию в формат .jpeg
    for clip_limit in np.arange(0.1, 10, 0.1):
        img_jpeg = apply_clahe_1(img, clip_limit, tile_generation)
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
        clahe_result_bgr = img_jpeg  #  исходное трехканальное изображение
        # text = f'clipLimit: {clip_limit:.2f}\nQualityGrad {quality_grad}\nQualityBRISQUE: {quality_brisque:.2f}\nQualityPSNR: {quality_psnr:.2f}\nQualitySSIM: {quality_ssim:.2f}'
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1  # Коэффициент для масштабирования шрифта
        # font_color = (100, 255, 255)
        # font_thickness = 2
        # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # text_x = 10
        # text_y = (height + text_size[1]) // 2
        # # Разбиваем текст на строки и пишем каждую строку отдельно
        # line_spacing = 5
        # lines = text.split('\n')
        # for i, line in enumerate(lines):
        #     cv2.putText(clahe_result_bgr, line, (text_x, text_y + i * (text_size[1] + line_spacing)), font, font_scale,
        #                 font_color, font_thickness)
        qualities.append(clip_limit)
        # Сохранение параметров
        quality_brisque_values_cl.append(quality_brisque)
        quality_psnr_values_cl.append(quality_psnr)
        quality_ssim_values_cl.append(quality_ssim)
        quality_grad_values.append(quality_grad)
        output_image_path = f'output_{clip_limit:.2f}_{quality_brisque:.2f}_{quality_psnr:.2f}_{quality_ssim:.2f}.jpg' # имя файла для сохранения
        cv2.imwrite(output_image_path, clahe_result_bgr)

        output_images.append((clahe_result_bgr, quality_brisque))
    for i in range(len(qualities)):
        print(f'clipLimit: {qualities[i]}, QualityBRISQUE: {quality_brisque_values_cl[i]}, '
              f'QualityPSNR: {quality_psnr_values_cl[i]}, QualitySSIM: {quality_ssim_values_cl[i]}, '
              f'QualityGrad: {quality_grad_values[i]}')