import cv2
import random
import numpy as np


def adjust_contrast_brightness(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, b)
    return dst


def motion_blur(img):
    img = np.array(img)
    min_kernel_size = 5
    max_kernel_size = 10
    max_rotate_angle = 5
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    angle = random.uniform(-max_rotate_angle, max_rotate_angle)
    matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel_size))
    motion_blur_kernel = cv2.warpAffine(
        motion_blur_kernel, matrix, (kernel_size, kernel_size)
    )
    motion_blur_kernel = motion_blur_kernel / kernel_size
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def gaussian_blur(img):
    img = np.array(img)
    min_kernel_size = 3
    max_kernel_size = 3
    min_sigma = 0.25
    max_sigma = 1.0
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    sigma = random.uniform(min_sigma, max_sigma)
    blurred = cv2.GaussianBlur(
        img, ksize=(kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma
    )
    return blurred
