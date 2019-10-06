import cv2
import numpy as np


class ImagesConverter:
    """Class equipped with basic image filters"""

    def __init__(self):
        pass

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
