import cv2
from matplotlib import image as mpimg


im1 = cv2.imread("p1.jpg")
cv2.imshow("image BGR", im1)
cv2.waitKey(0)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
cv2.imshow("image RGB", im1)
cv2.waitKey(0)
im2 = mpimg.imread("p2.jpg")

# XOR with CV2
result = cv2.bitwise_xor(im1, im2)

cv2.imshow("result image", result)
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray image", gray)
cv2.waitKey(0)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("edged image", edged)
cv2.waitKey(0)

# Finding Contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

filteredContours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 150 and h > 150:
        filteredContours.append(contour)
        cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 255, 0), 3)

# # Draw all contours
# cv2.drawContours(im1, filteredContours, -1, (0, 255, 0), 3)

im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
cv2.imshow("Contours", im1)
cv2.waitKey(0)
cv2.destroyAllWindows()
