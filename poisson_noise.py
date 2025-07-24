#importing libraries
import cv2      
import matplotlib.pyplot as plt
import numpy as np

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

# generate normal distribution numbers
def norm_dis(mean, std, size):
    height, width = size

    # generate random numbers
    rand1 = np.random.rand(*size)
    rand2 = np.random.rand(*size)
    
    # apply Box-Muller transform to generate a normal distribution
    norm = np.sqrt(-2 * np.log(rand1)) * np.cos(2 * np.pi * rand2)

    # scale by mean and standard deviation
    ans = mean + std * norm
    return ans

# import image and make it grayscale
img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png")
img = grayscale(img)

# create poisson noise of varying strength
poisson_noise = np.sqrt(img) * norm_dis(0, 1, img.shape)
poisson_noise_more = np.sqrt(img) * norm_dis(0, 5, img.shape)

# apply poisson noise to image
noisy_img = img + poisson_noise
more_noisy_img = img + poisson_noise_more

# plot images and their results
plt.figure(figsize=(10, 10))     
plt.subplot(1,3,1)
plt.title("Original Image")
plt.axis('off')
plt.imshow(img, cmap='gray')   

plt.subplot(1,3,2)
plt.title("Noisy Image")
plt.axis('off')
plt.imshow(noisy_img, cmap='gray')   

plt.subplot(1,3,3)
plt.title("Noisier Image")
plt.axis('off')
plt.imshow(more_noisy_img, cmap='gray')   

plt.show()