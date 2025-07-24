## Updated September 15, 2024

#importing libraries
import cv2      
import math
import matplotlib.pyplot as plt
import numpy as np

# import blurry and sharp image
img_blur = cv2.imread("/Users/jeslynyang/.Trash/data/defocused_blurred/0_IPHONE-SE_F.JPG") 
img_sharp = cv2.imread("/Users/jeslynyang/.Trash/data/sharp/0_IPHONE-SE_S.JPG")
img_blur = np.flip(img_blur, 2)
img_sharp = np.flip(img_sharp, 2)

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

# calculate variance of image (2D array)
def calc_var(arr):  
    means = [row.mean() for row in arr]
    square_diff = [abs(row - mean)**2 for row, mean in zip(arr, means)]
    vars = [row.mean() for row in square_diff]
    var = sum(vars) / len(vars)
    return var

# detect if an image is blurry
def detect_blur(img):
    #laplacian filter
    laplacian = np.array([[0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0]])  
    
    # apply laplacian filter to image and calculate variance after
    img_lap = pass_filter(img, laplacian)
    img_var = calc_var(img_lap)

    # check if an image has a variance higher than 100 threshold
    if img_var > 100:
        result = "Not blurry"
    else:
        result = "Blurry"

    return result, img_var

# detect if image is blurry or not
sharp_result, img_var_sharp = detect_blur(img_sharp)	
blur_result, img_var_blur = detect_blur(img_blur)	

# plot images and their results
plt.figure(figsize=(10, 10))     
plt.subplot(1,2,1)
img_var_sharp = round(img_var_sharp, 2)
plt.title(str(sharp_result) + ": " + str(img_var_sharp))
plt.imshow(img_sharp)   
plt.subplot(1,2,2)
img_var_blur = round(img_var_blur, 2)
plt.title(str(blur_result) + ": " + str(img_var_blur))
plt.imshow(img_blur)
plt.show()