import cv2
import numpy as np
import os
from skimage import io, color, util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def image_quality_squared_gradient(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    squared_gradient = gradient_x ** 2 + gradient_y ** 2
    quality = np.mean(squared_gradient)
    return quality
def calculate_brisque_quality(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    brisque_score = cv2.quality.QualityBRISQUE_compute(image, "brisque_model_live.yml", "brisque_range_live.yml")
    return brisque_score[0], blur_score
def calculate_PSNR_quality(image1, image2):
    if len(image1.shape) == 3:
        image1 = color.rgb2gray(image1)
    if len(image2.shape) == 3:
        image2 = color.rgb2gray(image2)
    mse = ((image1 - image2) ** 2).mean()
    psnr_value = 10 * (np.log10((255 ** 2) / (mse)))
    return psnr_value
def calculate_SSIM_quality(image1, image2):
    if len(image1.shape) == 3:
        image1 = color.rgb2gray(image1)
    if len(image2.shape) == 3:
        image2 = color.rgb2gray(image2)
    data_range = 1.0 if image1.dtype == np.float64 else 255
    ssim_value = ssim(image1, image2, data_range=data_range)
    return ssim_value
def process_images_in_folders(original_folder, distorted_folder, output_folder):
    quality_brisque_values = []
    quality_grad_values = []
    quality_psnr_values = []
    quality_ssim_values = []

    original_image_files = [f for f in os.listdir(original_folder) if
                            f.endswith(('.jpg', '.jpeg', '.png', '.BMP', '.bmp'))] # перебор всех изображений, находящихся в папке
    distorted_image_files = [f for f in os.listdir(distorted_folder) if
                             f.endswith(('.jpg', '.jpeg', '.png', '.BMP', '.bmp'))]

    orig_image_path = os.path.join(original_folder, original_image_files[0])
    orig_image = cv2.imread(orig_image_path)

    with open(os.path.join(output_folder, f"{os.path.basename(distorted_folder)}_metrics.txt"), 'w') as file:
        for dist_file in distorted_image_files:
            dist_image_path = os.path.join(distorted_folder, dist_file)
            dist_image = cv2.imread(dist_image_path)

            quality_brisque, _ = calculate_brisque_quality(dist_image)
            quality_grad = image_quality_squared_gradient(dist_image)
            quality_psnr = calculate_PSNR_quality(orig_image, dist_image)
            quality_ssim = calculate_SSIM_quality(orig_image, dist_image)

            height, width, channels = dist_image.shape  # получаем высоту, ширину и количество каналов изображения
            clahe_result_bgr = dist_image  # Берем исходное трехканальное изображение
            text = f'QualityBRISQUE: {quality_brisque:.2f}\nQualitySSIM: {quality_ssim:.2f}\nQualityGrad {quality_grad:.2f}\nQualityPSNR: {quality_psnr:.2f}\n'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Коэффициент для масштабирования шрифта
            font_color = (0, 255, 0)
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = (height + text_size[1]) // 2
            line_spacing = 5
            lines = text.split('\n')
            for i, line in enumerate(lines):
                cv2.putText(clahe_result_bgr, line,
                            (text_x, text_y + i * (text_size[1] + line_spacing)),
                            font, font_scale, font_color, font_thickness)
            quality_brisque_values.append(quality_brisque)
            quality_grad_values.append(quality_grad)
            quality_psnr_values.append(quality_psnr)
            quality_ssim_values.append(quality_ssim)

            output_path = os.path.join(output_folder, f"processed_{dist_file}")
            cv2.imwrite(output_path, dist_image)

            file.write(
                f'QualityBRISQUE: {quality_brisque:.2f}, SSIM: {quality_ssim:.2f}, QualityGrad: {quality_grad:.2f}, PSNR: {quality_psnr:.2f}\n')

if __name__ == "__main__":
    original_folder_path = "F://PycharmProjects//blur_salt-papper-jpg//reference_images"
    distorted_folder_path = "F://PycharmProjects//blur_salt-papper-jpg//distorted_images//airplane"
    output_folder_path = "F://PycharmProjects//blur_salt-papper-jpg//distorted_images//airplane//changes"
    os.makedirs(output_folder_path, exist_ok=True)
    process_images_in_folders(original_folder_path, distorted_folder_path, output_folder_path)
