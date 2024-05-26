import cv2
from matplotlib import image as mpimg


im1 = cv2.imread("p1.jpg")
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = mpimg.imread("p2.jpg")

# XOR with CV2
result = cv2.bitwise_xor(im1, im2)

cv2.imshow("result image", result)
cv2.waitKey(0)
