#importing libraries
import cv2    
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

# calculate ssim of image
def ssim(og_img, img, k1=0.01, k2=0.03):
    # mean of x and y
    avg_x = calc_avg(og_img)
    avg_y = calc_avg(img)

    # variance of x and y
    var_x = calc_var(og_img, avg_x)
    var_y = calc_var(img, avg_y)

    # covariance of x and y
    covar_xy = calc_covar(og_img, img, avg_x, avg_y)

    # dynamic range of pixel values
    L = 255

    # stablize division with weak denominator
    c1 = (k1*L)**2
    c2 = (k2*L)**2

    # calculate ssim according to equation
    score = ((2 * avg_x * avg_y + c1) * (2 * covar_xy + c2)) / ((avg_x**2 + avg_y**2 + c1) * (var_x + var_y + c2))

    return score

# calculate average of image (matrix)
def calc_avg(arr):
    # calculate mean
    avg = np.mean(arr)

    return avg

# calculate variance of image (matrix)
def calc_var(arr, mean):  
    # calculate variance
    var = np.mean((arr - mean) ** 2)

    return var

# calculate variance of image (matrix)
def calc_covar(arr1, arr2, mean_arr1, mean_arr2):  
    # calculate covariance 
    covar = np.mean((arr1 - mean_arr1) * (arr2 - mean_arr2))

    return covar

# process and grayscale original sharp image
img_sharp = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/sharp/304_SAMSUNG-GALAXY-J7_S.jpg")
img_sharp = np.flip(img_sharp, 2)
img_sharp = grayscale(img_sharp)

# process and grayscale original blurry image
img_blur = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/dataset/defocused_blurred/304_SAMSUNG-GALAXY-J7_F.jpg")
img_blur = grayscale(img_blur)

# calculate ssim between 2 images
ssim_sharp = ssim(img_sharp, img_sharp)
ssim_blur = ssim(img_sharp, img_blur)

# plot original sharp image
plt.figure(figsize=(10, 10))     
plt.subplot(1, 2, 1)
ssim_sharp = round(ssim_sharp, 2)
plt.title("Original Sharp\n" + "SSIM: " + str(ssim_sharp))
plt.axis("off")
plt.imshow(img_sharp, cmap='gray')    

# plot original blurry image   
plt.subplot(1, 2, 2)
ssim_blur = round(ssim_blur, 2)
plt.title("Original Blurry\n" + "SSIM: " + str(ssim_blur))
plt.axis("off")
plt.imshow(img_blur, cmap='gray')    
plt.show()