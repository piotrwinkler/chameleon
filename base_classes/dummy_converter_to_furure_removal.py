import cv2
import numpy as np

from base_classes.tester import Tester


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

    @staticmethod
    def sobel_filter(img, depth="8U", kernel_size=3):
        """
        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#sobel
        :param img: Input image (gray or RGB or BGR)
        :param depth:
        :param kernel_size:
        :return: Input image with Sobel filter
        """
        # img = ImagesConverter.rgb_to_gray(img) #====================================================
        if depth == "8U":
            img_depth = cv2.CV_8U
            img_sobel_x = cv2.Sobel(img, img_depth, 1, 0, ksize=kernel_size)
            img_sobel_y = cv2.Sobel(img, img_depth, 0, 1, ksize=kernel_size)
            # Tester.show_image(img_sobel_x + img_sobel_y)
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


class RgbtoGray:
    def __init__(self):
        pass

    def __call__(self, img):
        print(np.shape(img))
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class FilterImageSobelx:
    def __init__(self):
        pass

    def __call__(self, img):
        return cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)


class FilterImageSobely:
    def __init__(self):
        pass

    def __call__(self, img):
        return cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)


class NormalizeImage255:
    def __init__(self):
        pass

    def __call__(self, img):
        return img/255.0


class NormalizeImage:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img - np.min(img)
        img = img / np.max(img)
        return img


if __name__ == "__main__":
    img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/test_dataset/arab.jpg"
    img = cv2.imread(img_path).astype(np.float32)

    normalize_image255 = NormalizeImage255()
    rgb_to_gray = RgbtoGray()
    filter_image_sobelx = FilterImageSobelx()
    normalize_image = NormalizeImage()

    img = cv2.resize(img, (256, 256))
    img = normalize_image255(img)
    img = rgb_to_gray(img)
    # gray_img = ImagesConverter.normalize_image255(gray_img)
    filtered_img = filter_image_sobelx(img)

    filtered_img = np.array(filtered_img, dtype='float32')

    cv2.imshow(f'filtered_img', filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
