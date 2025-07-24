## Updated September 15, 2024

#importing libraries
import cv2     
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# grayscales an image and convert it to be a 2D array
def grayscale(img):
    height, width, color = img.shape
    gray_image = np.zeros((height, width)).astype(np.uint8) 

    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            # get R, G, B values of each pixel
            pixel = img[y][x]
            R,G,B = pixel

            # replace pixel with grayscale version
            gray_val = 0.2126*R + 0.7152*G + 0.0722*B  
            gray_image[y][x] = gray_val

    return gray_image

# make gaussian kernel for blur
def guassian_kernel(size, sigma):  
    # initialize number of rows and columns based on size
    numRows, numCols = size
    kernel = np.zeros((numRows, numCols), dtype=np.float32)

    # make gaussian kernel 
    for i in range(numRows):
        for j in range(numCols):
            # generate centered gaussian kernel according to equation
            x = i - numRows//2
            y = j - numCols//2

            kernel[i, j] = np.exp(-0.5 * ((x**2 + y**2) / (sigma**2)))
    return kernel

# process blurry version of the image
img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png")

# grayscale images and convert image to 2D array
img = grayscale(img)

# apply gaussian blur to image
kernel_size = img.shape
sigma = 25
guassian = guassian_kernel(kernel_size, sigma)

# convert image from spatial to fourier
ft = fft2(img)
ft = fftshift(ft)

# multiply image fourier transform by guassian kernel in fourier domain
img_blur = ft * guassian

# convert image from fourier to spatial
img_blur_spatial = ifftshift(img_blur)
img_blur_spatial = ifft2(img_blur_spatial)
img_blur_spatial = np.abs(img_blur_spatial)

# deconvolve with Weiner filter according to equation
k = 0.01
sharp_img = (np.conj(guassian) / (np.abs(guassian)**2 + k)) * img_blur

# convert image from fourier to spatial
sharp_img = ifft2(sharp_img)
sharp_img = np.abs(sharp_img)

# plot original image
plt.figure(figsize=(10, 10))     
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')  

# plot blurry images
plt.subplot(1,3,2)
plt.title("Blurred Image")
plt.imshow(img_blur_spatial, cmap='gray')   

# plot sharpened image
plt.subplot(1,3,3)
plt.title("Sharpened Image")
plt.imshow(sharp_img, cmap='gray')
plt.show()