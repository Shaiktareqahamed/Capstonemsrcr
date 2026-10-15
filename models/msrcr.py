import cv2
import numpy as np

def single_scale_retinex(img, sigma):
    return np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)

def msrcr_enhance(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0):
    img_float = np.float64(img) + 1.0
    msr = np.zeros_like(img_float)
    for sigma in sigmas:
        msr += single_scale_retinex(img_float, sigma)
    msr = msr / len(sigmas)
    img_sum = np.sum(img_float, axis=2, keepdims=True)
    cr = beta * (np.log10(alpha * img_float) - np.log10(img_sum))
    msrcr = msr * cr
    # Normalize cleanly without blowing out the white colors
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    msrcr = np.uint8(msrcr)
    # NEW: Apply an Unsharp Mask instead of the heavy denoiser
    # This brings back the sharp, crisp details of the objects
    gaussian_blur = cv2.GaussianBlur(msrcr, (0, 0), 2.0)
    sharp_img = cv2.addWeighted(msrcr, 1.5, gaussian_blur, -0.5, 0)
    return sharp_img