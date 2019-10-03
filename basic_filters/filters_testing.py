import cv2
import numpy as np


img_path = 'datasets/street/art379.jpg'
# img_path = 'datasets/samolot_mini_mini.jpg'

img = cv2.imread(img_path)
cv2.imshow("Original Image", img)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray", gray)

# canny (mean is the best)
mean_pixel = np.mean(img)
print("mean = {}".format(mean_pixel))
img_canny_mean = cv2.Canny(img, 0.6 * mean_pixel, 1.33 * mean_pixel)
cv2.imshow("Canny mean", img_canny_mean)

median_pixel = np.median(img)
print("median = {}".format(median_pixel))
img_canny_median = cv2.Canny(img, 0.6 * median_pixel, 1.33 * median_pixel)
cv2.imshow("Canny median", img_canny_median)

img_canny_hardcoded = cv2.Canny(img, 100, 200)
cv2.imshow("Canny hardcoded", img_canny_hardcoded)

# sobel
img_sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
img_sobel = img_sobel_x + img_sobel_y

sobel_x64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
abs_sobel_x64f = np.absolute(sobel_x64f)
sobel_8u_x = np.uint8(abs_sobel_x64f)

sobel_y64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
abs_sobel_y64f = np.absolute(sobel_y64f)
sobel_8u_y = np.uint8(abs_sobel_y64f)

cv2.imshow("Sobel 64FX", sobel_8u_x)
cv2.imshow("Sobel 64FY", sobel_8u_y)
cv2.imshow("Sobel 64F", sobel_8u_x + sobel_8u_y)

cv2.imshow("Sobel X", img_sobel_x)
cv2.imshow("Sobel Y", img_sobel_y)
cv2.imshow("Sobel", img_sobel)


# prewitt
kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
img_prewitt_x = cv2.filter2D(gray, -1, kernel_x)
img_prewitt_y = cv2.filter2D(gray, -1, kernel_y)
cv2.imshow("Prewitt X", img_prewitt_x)
cv2.imshow("Prewitt Y", img_prewitt_y)
cv2.imshow("Prewitt", img_prewitt_x + img_prewitt_y)

# Roberts cross
kernel_x = np.array([[1, 0], [0, -1]])
kernel_y = np.array([[0, 1], [-1, 0]])
img_roberts_cross_x = cv2.filter2D(gray, -1, kernel_x)
img_roberts_cross_y = cv2.filter2D(gray, -1, kernel_y)
cv2.imshow("Roberts cross X", img_roberts_cross_x)
cv2.imshow("Roberts cross Y", img_roberts_cross_y)
cv2.imshow("Roberts cross", img_roberts_cross_x + img_roberts_cross_y)


cv2.waitKey(0)
cv2.destroyAllWindows()
