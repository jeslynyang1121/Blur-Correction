#importing libraries
import cv2      
import numpy as np
from scipy.fft import fft2, ifft2
import ssim
import PSNR
import math
import matplotlib.pyplot as plt

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

# normalizes an image
def normalize(img):
    # ensure image values are in range
    normal_img = np.clip(img, 0, None)

    # normalize matrix
    normal_img /= normal_img.sum()
    # min_val = np.min(img)
    # max_val = np.max(img)
    # normal_img = ((normal_img - min_val) / (max_val - min_val)) * 255.0
    
    return normal_img

# Gaussian Blur PSF creation
def gaussBlur(sigma, imageSize):
    height, width = imageSize
    size = math.ceil(sigma) * 3 * 2 + 1

    # Temp array of matrix for PSF
    psf = np.zeros((size, size), dtype='float64')

    center = size // 2

    # Build PSF pixel by pixel
    for i in range(size):
        for j in range(size):
            # Takes distance from center of matrix (x,y)
            x = j - center
            y = i - center
            # Calculate PSF through Gaussian distribution exquation
            psf[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**2)

    # Normalize PSF
    psf = normalize(psf)

    # Pads PSF within image specifics
    pad_height  = (height - size) // 2
    pad_width  = (width - size) // 2

    padding = ((pad_height, height - size - pad_height), 
                (pad_width, width - size - pad_width))
    padded_psf = np.pad(psf, padding, mode='constant')

    return padded_psf

# Use fourier transforms to convolve the image
def fft_convolve(img, psf):
    # convert image and PSF to fourier transforms
    img_fft = fft2(img)
    psf_fft = fft2(np.fft.ifftshift(psf))
    conv_fft = img_fft * psf_fft

    # convert image and PSF back to image
    conv_img = np.real(ifft2((conv_fft)))

    return conv_img

# richardson-lucy algorithm for calculating PSF
def richardson_lucy_psf(img, restored_img, psf, num_iter):
    for i in range(num_iter):
        # calculate PSF according to equation
        blur_img = fft_convolve(restored_img, psf) 
        blur = img / (blur_img + 1e-10)
        deconvolve = fft_convolve(blur, restored_img[::-1, ::-1])   # flip image with [::-1, ::-1]
        psf *= deconvolve

        # normalize PSF
        psf = normalize(psf)

    return psf

# richardson-lucy algorithm for calculating restored image
def richardson_lucy_img(img, restored_img, psf, num_iter):
    for i in range(num_iter):
        # calculate restored image according to equation
        blur_img = fft_convolve(restored_img, psf) 
        blur = img / (blur_img + 1e-10) 
        deconvolve = fft_convolve(blur, psf[::-1, ::-1])    # flip PSF with [::-1, ::-1]
        restored_img *= deconvolve

        # normalize image
        restored_img = normalize(restored_img)

    return restored_img

def brighten(img):
    # calculate scale of values
    scale = 5
    maxVal = np.average(img)
    while (maxVal < 1):
        scale *= 10
        maxVal *= 10

    # scale image so values are in the 1's place
    img = (img * scale).astype('float64')

    # manually brighten by constant value
    img += 240 - np.average(img)
    return img

# nonblind version of richardson-lucy algorithm
def richardson_lucy(img, num_inner, num_outer):

    # initialize guess for restored image and psf
    restored_img = np.copy(img).astype(np.float64)
    psf = gaussBlur(13, img.shape)
    
    # perform Richardson-Lucy algorithm num_outer times
    for i in range(num_outer):
        # calculate new PSF and restored image num_inner times
        psf = richardson_lucy_psf(img, restored_img, psf, num_inner)
        restored_img = richardson_lucy_img(img, restored_img, psf, num_inner)
    
    # add value to adjust brightness of image
    restored_img = brighten(restored_img)

    # ensure image only contains acceptable values in range (0, 255)
    restored_img = np.clip(restored_img, 0, 255)

    return restored_img 

# passes filter through image
def pass_filter(img, filter):  
    # make an empty image
    new_image = np.zeros(np.shape(img), dtype=np.int64) 

    size_offset = int(len(filter)/2)

    # Iterate through each pixel                                                   
    for y in range(size_offset, img.shape[0] - size_offset):        
        for x in range(size_offset, img.shape[1] - size_offset):  
            # apply laplacian filter to image, 3x3 section at a time
            channel_subset = img[y-size_offset : y+size_offset+1, x-size_offset : x+size_offset+1]
            new_image[y][x] = int(np.sum(np.multiply(filter, channel_subset)))
    return new_image

# calculate variance of image (matrix)
def calc_var(arr):  
    # calculate mean and square difference of rows
    means = [row.mean() for row in arr]
    square_diff = [abs(row - mean)**2 for row, mean in zip(arr, means)]

    # calculate variance
    vars = [row.mean() for row in square_diff]
    var = sum(vars) / len(vars)
    return var

# calculate similarity between orginal image and image
def calc_similarity(og_img, img):
    ssim_val = ssim.ssim(og_img, img)

    mse_val = PSNR.MSE(og_img, img)
    psnr_val = PSNR.PSNR(mse_val)

    return ssim_val, psnr_val

# detect if an image is blurry
def detect_blur(img):
    #laplacian filter
    laplacian = np.array([[0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0]])  
    
    # apply laplacian filter to image and calculate variance after
    img_lap = pass_filter(img, laplacian)
    img_var = calc_var(img_lap)

    # check if an image has a variance higher than 100 threshold to conclude blurriness
    if img_var > 100:
        result = "Not blurry"
    else:
        result = "Blurry"

    return result, img_var

# process and grayscale original sharp image
# img_sharp = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/sharp/322_HONOR-7C_S.jpg")
img_sharp = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/sharp/188_NIKON-D3400-18-55MM_S.JPG")
# img_sharp = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/sharp/304_SAMSUNG-GALAXY-J7_S.jpg")
img_sharp = grayscale(img_sharp)

# detect if now blurry image is blurry
# _, img_var_sharp = detect_blur(img_sharp)
img_var_sharp = calc_similarity(img_sharp, img_sharp)

# process and grayscale original blurry image
# img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/defocused_blurred/322_HONOR-7C_F.jpg")
img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/defocused_blurred/188_NIKON-D3400-18-55MM_F.JPG")
# img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/defocused_blurred/304_SAMSUNG-GALAXY-J7_F.jpg")
img = grayscale(img)
ssim_og_img, psnr_og_img = calc_similarity(img, img)

# make image blurrier if necassary
blur = gaussBlur(12, img.shape)
img = fft_convolve(img, blur)

# detect if now blurry image is blurry
# _, img_var_blur = detect_blur(img)
ssim_blurry_img, psnr_blurry_img = calc_similarity(img_sharp, img)

# perform richardson lucy algorithm in varying strengths
sharp_img1 = richardson_lucy(img, 10, 20)
sharp_img2 = richardson_lucy(img, 20, 20)

# detect if now sharpened images are blurry
# _, sharp_img1_var = detect_blur(sharp_img1)
# _, sharp_img2_var = detect_blur(sharp_img2)
ssim_sharp_img1, psnr_sharp_img1 = calc_similarity(img_sharp, sharp_img1)
ssim_sharp_img2, psnr_sharp_img2 = calc_similarity(img_sharp, sharp_img2)

# plot original sharp image
plt.figure(figsize=(10, 10))     
plt.subplot(1, 4, 1)
ssim_og_img = round(ssim_og_img, 2)
psnr_og_img = round(psnr_og_img, 2)
plt.title("Original Sharp\n" + "SSIM: " + str(ssim_og_img) + "\nPSNR: " + str(psnr_og_img))
# plt.title("Original Sharp")
plt.axis("off")
plt.imshow(img_sharp, cmap='gray')    

# plot original blurry image   
plt.subplot(1, 4, 2)
ssim_blurry_img = round(ssim_blurry_img, 2)
psnr_blurry_img = round(psnr_blurry_img, 2)
plt.title("Original Blurry\n" + "SSIM: " + str(ssim_blurry_img) + "\nPSNR: " + str(psnr_blurry_img))
plt.axis("off")
plt.imshow(img, cmap='gray')    

# plot sharpened image
plt.subplot(1, 4, 3)
ssim_sharp_img1 = round(ssim_sharp_img1, 2)
psnr_sharp_img1 = round(psnr_sharp_img1, 2)
plt.title("10 Inner + 20 Outer\n" + "SSIM: " + str(ssim_sharp_img1) + "\nPSNR: " + str(psnr_sharp_img1))
# plt.title("10 Inner + 20 Outer")
plt.axis("off")
plt.imshow(sharp_img1, cmap='gray')

# plot sharpened image
plt.subplot(1, 4, 4)
ssim_sharp_img2 = round(ssim_sharp_img2, 2)
psnr_sharp_img2 = round(psnr_sharp_img2, 2)
plt.title("20 Inner + 20 Outer\n" + "SSIM: " + str(ssim_sharp_img2) + "\nPSNR: " + str(psnr_sharp_img2))
# plt.title("20 Inner + 20 Outer")
plt.axis("off")
plt.imshow(sharp_img2, cmap='gray')
plt.show()