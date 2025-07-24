## Updated September 5, 2024

# import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image
image_og = cv2.imread("./hopper.jpeg")
image = np.flip(image_og, 2)

def pass_filter(img, filter1, filter2):
    # make an empty image
    new_image = np.zeros(np.shape(img), dtype=np.int64) 

    size_offset = int(len(filter1)/2)
    
    # Iterate through each color in each pixel
    for y in range(size_offset, img.shape[0] - size_offset):        
        for x in range(size_offset, img.shape[1] - size_offset):    
            for c in range(img.shape[2]):                           
                # apply sobel x and y operators to image, 3x3 section at a time
                channel_subset = img[y-size_offset : y+size_offset+1, x-size_offset : x+size_offset+1, c]

                output_x = int(np.sum(np.multiply(filter1, channel_subset)))
                output_y = int(np.sum(np.multiply(filter2, channel_subset)))

                # combine x and y results before normalization
                new_image[y][x][c] = np.sqrt(np.square(output_x) + np.square(output_y))

    # Normalize colors in new image to a 0-1 range
    minval = np.min(new_image)
    maxval = np.max(new_image - minval)
    if (maxval != 0):
        new_image = ((new_image - minval) / maxval)
    return new_image

# Sobel operator masks, in x and y
sobel_y = np.array([[  1,  2,  1],
                    [  0,  0,  0],
                    [ -1, -2, -1]])
sobel_x = np.array([[  -1,  0, 1],
                    [  -2,  0, 2],
                    [  -1,  0, 1]])

# Apply Sobel edge detection on image
output_img = pass_filter(image, sobel_x, sobel_y)

# display image before and after Sobel detection
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Sobel Edge Detection")
plt.imshow(output_img)
plt.show()


