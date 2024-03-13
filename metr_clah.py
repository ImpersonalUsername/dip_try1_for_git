import cv2
import numpy as np
from skimage import io, color, util
from skimage.metrics import structural_similarity as ssim

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