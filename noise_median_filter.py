## Updated September 15, 2024

# import libraries
import cv2
import random
import numpy as np
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

# add salt and pepper noise to an image
def add_sp_noise(img, salt_prob, pepper_prob):
    rows, cols = img.shape
    noisy_img = img.copy()
    
    # Loop over through every pixel in the image
    for i in range(rows):
        for j in range(cols):
            val = random.random()       # Generate a random value between 0 and 1

            # Add salt noise (white pixel)
            if val < salt_prob:
                noisy_img[i, j] = 255   # Set pixel to white
           
            # Add pepper noise (black pixel)
            elif val > (1 - pepper_prob):
                noisy_img[i, j] = 0     # Set pixel to black

    return noisy_img

# apply the median filter to an image
def median_filter(img):
    rows, cols = img.shape
    clear_img = img.copy()      # duplicate and change noisy image

    # Loop over through every pixel in the image
    for x in range(rows):
        for y in range(cols):
            around = []

            # calculate median over 3x3 around pixel
            for i in range (max(0, x-1), min(x+1, rows)):
                for j in range (max(0, y-1), min(y+1, cols)):
                    around.append(img[i, j])
            
            index = len(around) // 2
            around.sort()
            med = around[index]
            
            # set pixel to be median
            clear_img[x, y] = med

    return clear_img

# Load image
imgPath = '/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png'
img = cv2.imread(imgPath) 

# grayscale an image
img = grayscale(img)

# Add salt and pepper noise to image
salt_prob = 0.1     # Probability of salt noise
pepper_prob = 0.1   # Probability of pepper noise
noisy_img = add_sp_noise(img, salt_prob, pepper_prob)

# apply median filter to reduce noise
clear_img = median_filter(noisy_img)

# original image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap = "gray")
plt.axis("off")

# noisy image 
plt.subplot(1, 3, 2)
plt.title("Salt & Pepper Noisy Image")
plt.imshow(noisy_img, cmap= "gray")
plt.axis("off")

# clear image 
plt.subplot(1, 3, 3)
plt.title("Clear Image")
plt.imshow(clear_img, cmap= "gray")
plt.axis("off")

plt.show()


