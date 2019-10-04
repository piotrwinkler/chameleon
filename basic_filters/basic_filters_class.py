import cv2
import numpy as np


class ImagesConverter:
    """Class equipped with basic image filters"""

    def __init__(self):
        pass

    @staticmethod
    def bgr_to_rgb(img):
        """
        :param img: BGR image
        :return: RGB image
        """
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img

    @staticmethod
    def rgb_to_bgr(img):
        """
        :param img: RGB image
        :return: BGR image
        """
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return bgr_img

    @staticmethod
    def bgr_to_gray(img):
        """
        :param img: BGR image
        :return: gray image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def rgb_to_gray(img):
        """
        :param img: RGB image
        :return: gray image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    @staticmethod
    def canny_filter(img, threshold_type, threshold_lower=100, threshold_upper=200):
        """
         https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
        :param img: BGR image
        :param threshold_type:
        :param threshold_lower:
        :param threshold_upper:
        :return: BGR image with applied Canny filter
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

    @staticmethod
    def gaussian_blur(img, kernel_size=(3, 3)):
        """
        Smoothing image and reducing noise.
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur

        :param img: BGR image
        :param kernel_size: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
        :return: BGR image blurs with Gaussian filter
        """

        img_gaussian = cv2.GaussianBlur(img, kernel_size, 0)
        return img_gaussian

    @staticmethod
    def sobel_filter(img, depth="8U", kernel_size=3):
        """
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#sobel
        :param img: Input image (gray or RGB or BGR)
        :param depth:
        :param kernel_size:
        :return: Input image with Sobel filter
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

    @staticmethod
    def prewitt_filter(img):
        """
        :param img: Input image (gray or RGB or BGR)
        :return: Input image with Prewitt filter
        """
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
        img_prewitt_y = cv2.filter2D(img, -1, kernel_y)
        return img_prewitt_x + img_prewitt_y

    @staticmethod
    def roberts_cross_filter(img):
        """
        :param img: Input image (gray or RGB or BGR)
        :return: Input image with Roberts Cross filter
        """
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        img_roberts_cross_x = cv2.filter2D(img, -1, kernel_x)
        img_roberts_cross_y = cv2.filter2D(img, -1, kernel_y)
        return img_roberts_cross_x + img_roberts_cross_y

    @staticmethod
    def bgr_to_sephia(img):
        """
        :param img: BGR image
        :return: Sephia image
        """
        kernel = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]])

        img_sepia = cv2.transform(img, kernel)
        # Check which entries have a value greater than 255 and set it to 255
        img_sepia[np.where(img_sepia > 255)] = 255
        return img_sepia

    @staticmethod
    def rgb_to_sephia(img):
        """
        :param img: RGB image
        :return: Sephia image
        """
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])

        img_sepia = cv2.transform(img, kernel)
        # Check which entries have a value greater than 255 and set it to 255
        img_sepia[np.where(img_sepia > 255)] = 255

        return img_sepia

    @staticmethod
    def negative_filter(img):
        """
        :param img: Input image (BGR or RGB)
        :return: Negative image
        """
        negative_img = 255 - img
        return negative_img


    @staticmethod
    def show_single_img(img, title="Image"):
        """
        Display BGR image and wait for key
        :param img: BGR image
        :param title: plot title
        :return: None
        """
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_image_without_waitkey(img, title="Image"):
        """
        Display BGR image and continue process
        :param img: BGR image
        :param title: plot title
        :return: None
        """
        cv2.imshow(title, img)

    @staticmethod
    def load_img(load_path):
        """
        Load image to BGR format
        :param load_path: path to image
        :return: BGR image
        """
        return cv2.imread(load_path)

    @staticmethod
    def save_img(save_path, img):
        """
        Save BGR image
        :param save_path: path where to save img
        :param img: BGR image
        :return: None
        """
        cv2.imwrite(save_path, img)
