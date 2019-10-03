import cv2
import numpy as np


class BasicFilters(object):
    """Class equipped with basic image filters"""

    def __init__(self):
        pass

    def rgb_to_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    def canny_filter(self, img, threshold_type, threshold_lower=100, threshold_upper=200):
        """
        https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
        """
        if threshold_type == "mean":
            mean_pixel = np.mean(img)
            canny_img = cv2.Canny(img, 0.6 * mean_pixel, 1.33 * mean_pixel)
        elif threshold_type == "median":
            median_pixel = np.median(img)
            canny_img = cv2.Canny(img, 0.6 * median_pixel, 1.33 * median_pixel)
        elif threshold_type == "manual":
            canny_img = cv2.Canny(img, threshold_lower, threshold_upper)
        else:
            return None

        return canny_img

    def gaussian_blur(self, img, kernel_size=(3, 3)):
        """
        Smoothing image and reducing noise.
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur

        :param img: input image
        :param kernel_size: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
        :return: image blurs with Gaussian filter
        """

        img_gaussian = cv2.GaussianBlur(img, kernel_size, 0)
        return img_gaussian

    def sobel_filter(self, img, depth="8U", kernel_size=3):
        """
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#sobel
        """
        if depth == "8U":
            img_depth = cv2.CV_8U
            img_sobel_x = cv2.Sobel(img, img_depth, 1, 0, ksize=kernel_size)
            img_sobel_y = cv2.Sobel(img, img_depth, 0, 1, ksize=kernel_size)
            return img_sobel_x + img_sobel_y

        elif depth == "16U":
            img_depth = cv2.CV_16U
            """Not implemented"""
            return None

        elif depth == "16S":
            img_depth = cv2.CV_16S
            """Not implemented"""
            return None

        elif depth == "32F":
            img_depth = cv2.CV_32F
            """Not implemented"""
            return None

        elif depth == "64F":
            img_depth = cv2.CV_64F
            sobel_x64f = cv2.Sobel(img, img_depth, 1, 0, ksize=kernel_size)
            abs_sobel_x64f = np.absolute(sobel_x64f)
            sobel_8u_x = np.uint8(abs_sobel_x64f)

            sobel_y64f = cv2.Sobel(img, img_depth, 0, 1, ksize=kernel_size)
            abs_sobel_y64f = np.absolute(sobel_y64f)
            sobel_8u_y = np.uint8(abs_sobel_y64f)
            return sobel_8u_x + sobel_8u_y

        else:
            raise Exception('Wrong depth')
            return None

    def prewitt_filter(self, img):
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
        img_prewitt_y = cv2.filter2D(img, -1, kernel_y)
        return img_prewitt_x + img_prewitt_y

    def roberts_cross_filter(self, img):
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        img_roberts_cross_x = cv2.filter2D(img, -1, kernel_x)
        img_roberts_cross_y = cv2.filter2D(img, -1, kernel_y)
        return img_roberts_cross_x + img_roberts_cross_y

    def show_single_img(self, img, title="Image"):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image_without_waitkey(self, img, title="Image"):
        cv2.imshow(title, img)

    def load_img(self, load_path):
        return cv2.imread(load_path)

    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)
