import math as m
from pathlib import Path
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

def greyscaling(path):
    #takes image input, reads through each color channel, and converts it to greyscale.
    input_image = imread(Path(path)).astype("uint8")
    r, g, b = input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722
    grayscale_image = r_const * r + g_const * g + b_const * b
    return grayscale_image

def gaussian_noise(image):
    #Generating noise over a random distribution for testing. 
    gaussian_gen = np.random.normal(0, gaussian_intensity, image.shape)
    gaussian_image = image + gaussian_gen
    return gaussian_image

def MSE(greyscale_image, noisy_image):
    #calculating the mean of the difference in the image.
    mse_value = np.mean((greyscale_image - noisy_image) ** 2)
    return mse_value

def PSNR(mse_value):
    #Checking if the MSE is 0 to avoid dividing by 0.
    if mse_value == 0:
        PSNR_value = 100
        return PSNR_value
    else:
    #Calculating the PSNR value, since its greyscale, the max value is 25.
        PSNR_value = 20 * m.log10(255 / (m.sqrt(mse_value)))
        return PSNR_value

#Calling the assorted functions. 
greyscale_image = greyscaling("/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png")
gaussian_intensity = 30
noisy_image = gaussian_noise(greyscale_image)

#Calculating the PSNR for the original image. This should round to 0
MSE_value = MSE(greyscale_image, greyscale_image) 
PSNR_value = PSNR(MSE_value) 

#Calculating the PSNR for the noisy image. 
MSE_value_noisy = MSE(greyscale_image, noisy_image) 
PSNR_value_noisy = PSNR(MSE_value_noisy)  


#Showing the images. 
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(greyscale_image, cmap='gray')
plt.title(f"PSNR: {PSNR_value:.3f}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f"PSNR: {PSNR_value_noisy:.3f}")
plt.axis('off')
plt.show()
