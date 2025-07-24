## Updated September 15, 2024

#importing libraries
import cv2     
import matplotlib.pyplot as plt
import numpy as np
import ssim
import PSNR

# import blurry and sharp image
og_img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png")

img_sharp = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_tiny_blur.png")
img_sharp = np.flip(img_sharp, 2)

#laplacian filter
laplacian = np.array([[0, 1, 0],
                    [ 1,-4, 1],
                    [ 0, 1, 0]]) 

# grayscales an image
def grayscale(img):
    gray_image = np.zeros(np.shape(img)).astype(np.uint8) 

    # Iterate through each pixel
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # get R, G, B values of each pixel
            pixel = img[y][x]
            R,G,B = pixel

            # replace pixel with grayscale version
            gray_val = 0.2126*R + 0.7152*G + 0.0722*B  
            gray_image[y][x] = [gray_val, gray_val, gray_val] 

    return gray_image

# passes filter through image
def pass_filter(img, filter):  
    # make an empty image
    new_image = np.zeros(np.shape(img), dtype=np.int64) 

    size_offset = int(len(filter)/2)

    # Iterate through each color in each pixel                                                   
    for y in range(size_offset, img.shape[0] - size_offset):        
        for x in range(size_offset, img.shape[1] - size_offset):    
            for c in range(img.shape[2]):                           
                # apply laplacian filter to image, 3x3 section at a time
                channel_subset = img[y-size_offset : y+size_offset+1, x-size_offset : x+size_offset+1, c]
                new_image[y][x][c] = int(np.sum(np.multiply(filter, channel_subset)))
    return new_image

# pass median filter through image
def median_filter(img):  
    # make an empty image
    new_image = np.zeros(np.shape(img), dtype=np.int64) 

    size_offset = 3

    # Iterate through each color in each pixel                                                   
    for y in range(size_offset, img.shape[0] - size_offset):        
        for x in range(size_offset, img.shape[1] - size_offset):                         
            # apply median filter to image, 3x3 section at a time
            channel_subset = img[y-size_offset : y+size_offset+1, x-size_offset : x+size_offset+1]
            new_image[y][x] = np.median(channel_subset)
    return new_image

# normalizes an image
def normalize(img):				
    minval = np.min(img)                      
    maxval = np.max(img - minval)
    if (maxval != 0):
        img = ((img - minval) / maxval)
    return img

# calculate variance of image (2D array)
def calc_var(arr):  
    means = [row.mean() for row in arr]
    square_diff = [abs(row - mean)**2 for row, mean in zip(arr, means)]
    vars = [row.mean() for row in square_diff]
    var = sum(vars) / len(vars)
    return var

# applies an unsharp mask onto image
def unsharp_mask(img, strength):
    # standard formula for unsharp masking: 
    img_lap = pass_filter(img, laplacian)
    img_unsharp = np.zeros(np.shape(img)).astype(np.uint8) 

    for y in range(img.shape[0]): 
        for x in range(img.shape[1]):                       
            # apply laplacian filter to image, 3x3 section at a time
            img_unsharp[y][x] = img[y][x] - (img_lap[y][x] * strength)

            if img_unsharp[y][x][0] <= 0 or img_unsharp[y][x][0] >= 255:
                img_unsharp[y][x] = img[y][x]

    return img_unsharp

# detect if an image is blurry
def detect_blur(img): 
    
    # apply laplacian filter to image and calculate variance after
    img_lap = pass_filter(img, laplacian)
    img_var = calc_var(img_lap)

    # check if an image has a variance higher than 100 threshold
    if img_var > 100:
        result = "Not blurry"
    else:
        result = "Blurry"

    return result, img_var

# grayscale images
og_img = grayscale(og_img)
img_blur = grayscale(img_sharp)

# apply median filter and unsharp mask onto image
img_sharpen = unsharp_mask(img_blur, 0.7)

# detect if image is blurry or not
sharp_result, img_var_sharp = detect_blur(img_sharp)	

# calculate SSIM values 
ssim_og = ssim.ssim(og_img, og_img)
ssim_blur = ssim.ssim(og_img, img_blur)
ssim_sharp = ssim.ssim(og_img, img_sharpen)

# calculate PSNR values 
mse_og = PSNR.MSE(og_img, og_img)
psnr_og = PSNR.PSNR(mse_og)
mse_blur = PSNR.MSE(og_img, img_blur)
psnr_blur = PSNR.PSNR(mse_blur)
mse_sharp = PSNR.MSE(og_img, img_sharpen)
psnr_sharp = PSNR.PSNR(mse_sharp)

# plot original image
plt.figure(figsize=(10, 10))     
plt.subplot(1,3,1)
ssim_og = round(ssim_og, 2)
psnr_og = round(psnr_og, 2)
plt.title("Original\nSSIM: " + str(ssim_og) + "\nPSNR: " + str(psnr_og))
plt.axis('off')
plt.imshow(og_img) 

# plot blurry image   
plt.subplot(1,3,2)
ssim_blur = round(ssim_blur, 2)
psnr_blur = round(psnr_blur, 2)
plt.title("Blurry\nSSIM: " + str(ssim_blur) + "\nPSNR: " + str(psnr_blur))
plt.axis('off')
plt.imshow(img_blur)  
 
 # plot sharpened image
plt.subplot(1,3,3)
ssim_sharp = round(ssim_sharp, 2)
psnr_sharp = round(psnr_sharp, 2)
plt.title("Sharpened\nSSIM: " + str(ssim_sharp) + "\nPSNR: " + str(psnr_sharp))
plt.axis('off')
plt.imshow(img_sharpen)
plt.show()