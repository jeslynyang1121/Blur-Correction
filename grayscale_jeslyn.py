## Updated September 5, 2024

import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image and make an empty image for grayscale
image = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/Blur_F24/data/sharp/0_IPHONE-SE_S.JPG")
gray_image = np.zeros(np.shape(image)).astype(np.uint8) 

# Iterate through each pixel
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        # get R, G, B values of each pixel
        pixel = image[y][x]
        R,G,B = pixel

        # replace pixel with grayscale version
        gray_val = 0.2126*R + 0.7152*G + 0.0722*B  
        gray_image[y][x] = [gray_val, gray_val, gray_val] 

# display image
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(1,2,2)
plt.title("Gray Image")
plt.imshow(gray_image) 
plt.show()