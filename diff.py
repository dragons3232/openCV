import cv2
import numpy as np
from matplotlib import pyplot as plt, image as mpimg


im1 = cv2.imread('p1.jpg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = mpimg.imread("p2.jpg")

# Create figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2)

# Display the first image with a label
axes[0].imshow(im1)
axes[0].set_title('Image 1')
axes[0].axis('off')
# Display the second image with a label
axes[1].imshow(im2)
axes[1].set_title('Image 2')
axes[1].axis('off')

plt.show()


# Make into Numpy arrays
im1np = np.array(im1)*255
im2np = np.array(im2)*255

# XOR with Numpy
result = np.bitwise_xor(im1np, im2np).astype(np.uint8)

cv2.imshow('result image', result)
cv2.waitKey(0)

# plt.imshow(result)
# plt.show()
