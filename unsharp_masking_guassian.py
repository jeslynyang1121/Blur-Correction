import cv2      #importing libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import ssim

def mod(n, m):
        return ((n%m)+m)%m

def pass_filter(img, filter):
    new_image = np.zeros(np.shape(img), dtype=np.int64) # create new black image

    size_offset = int(len(filter)/2)
                                                                    # Loop for every...
    for y in range(img.shape[0]):        # row
        for x in range(img.shape[1]):    # pixel in row
            for c in range(img.shape[2]):                           # color in pixel

                y_min = max(0, y - size_offset)
                y_max = min(img.shape[0], y + size_offset + 1)
                x_min = max(0, x - size_offset)
                x_max = min(img.shape[1], x + size_offset + 1)

                channel_subset = img[y_min:y_max, x_min:x_max, c]       #apply filter, handle edges by mirroring to prevent black edges

                if y - size_offset < 0:         
                    channel_subset = np.vstack((np.flipud(channel_subset[:size_offset, :]), channel_subset))
                if y + size_offset >= img.shape[0]:
                    channel_subset = np.vstack((channel_subset, np.flipud(channel_subset[-size_offset:, :])))

                if x - size_offset < 0:
                    channel_subset = np.hstack((np.fliplr(channel_subset[:, :size_offset]), channel_subset))
                if x + size_offset >= img.shape[1]:
                    channel_subset = np.hstack((channel_subset, np.fliplr(channel_subset[:, -size_offset:])))

                new_image[y][x][c] = int(np.sum(np.multiply(filter, channel_subset)))
    return new_image


def normalize(new_image):				#this is only for displaying the image
    minval = np.min(new_image)                      
    maxval = np.max(new_image - minval)
    if (maxval != 0):
        new_image = ((new_image - minval) / maxval)
    return new_image

def normal(new_image):				#this is only for displaying the image, divides by 
    new_image = new_image / 256
    return new_image

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

img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_tiny_blur.png") #upload images, then make color adjustments
img = np.flip(img, 2)

gaussian = np.array([[0, 1, 0],\
                    [ 1, 1, 1],\
                    [ 0, 1, 0]])     #laplacian

blurred = pass_filter(img, gaussian)
sharp = img + (img - (blurred/5))*5
sharp = normal(sharp)

og_img = cv2.imread("/Users/jeslynyang/Desktop/cs/VIP/cat_clear.png")
og_img = grayscale(og_img)
img = grayscale(img)
sharp = grayscale(sharp)

ssim_blur = ssim.ssim(og_img, img)
ssim_sharp = ssim.ssim(og_img, sharp)

plt.figure(figsize=(16, 4))     #make figure with 3 spots
plt.subplot(1,2,1)
ssim_blur = round(ssim_blur, 2)
plt.title("Similarity: " + str(ssim_blur))
plt.imshow(img)

plt.subplot(1,2,2)
ssim_sharp = round(ssim_sharp, 2)
plt.title("Similarity: " + str(ssim_sharp))
plt.imshow(sharp)
plt.show()