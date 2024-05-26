import cv2
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.patches import ConnectionPatch


im1 = cv2.imread("p1.jpg")
im2 = cv2.imread("p2.jpg")

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
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 3)

# # Draw all contours
# cv2.drawContours(im1, filteredContours, -1, (0, 255, 0), 3)

# Draw lines between points on the two images
fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax2 = axes[1]

ax1.imshow(im1)
ax1.set_title("Image 1")
ax1.axis("off")
ax2.imshow(im2)
ax2.set_title("Image 2")
ax2.axis("off")

for cnt in filteredContours:
    x, y, w, h = cv2.boundingRect(cnt)
    points = [(x, y), (x + w, y + h)]
    for point in points:
        xy = point
        con = ConnectionPatch(
            xyA=xy,
            xyB=xy,
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            color="green",
        )
        ax2.add_artist(con)

plt.show()

cropped = im1[y : y + h, x : x + w]
cv2.imwrite("cropped.jpg", cropped)

im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
cv2.imshow("Contours", im1)
cv2.waitKey(0)
cv2.destroyAllWindows()
