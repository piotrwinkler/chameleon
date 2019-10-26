"""Setup creator use this script to get all conversions mentioned in "training_parameters.json" or in
"test_parameters.json" """
import cv2
import numpy as np


class FilterSharpen:
    def __init__(self):
        self._kernel = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])

    def __call__(self, img_):
        return cv2.filter2D(img_, -1, self._kernel)


class RgbtoGray:
    def __init__(self):
        pass

    def __call__(self, img_):
        return cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY).astype('float32')


class FilterImageSobelx:
    def __init__(self):
        pass

    def __call__(self, img_):
        return cv2.Sobel(img_, cv2.CV_32F, 1, 0, ksize=3).astype('float32')


class FilterImageSobely:
    def __init__(self):
        pass

    def __call__(self, img_):
        return cv2.Sobel(img_, cv2.CV_32F, 0, 1, ksize=3).astype('float32')


class Sepia:
    def __init__(self):
        self._kernel = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]])

    def __call__(self, img_):
        img_sepia = cv2.transform(img_, self._kernel)
        # Check which entries have a value greater than 255 and set it to 255
        img_sepia[np.where(img_sepia > 255)] = 255
        return img_sepia


class FilterCanny:
    def __init__(self, threshold_type='mean'):
        self._threshold_type = threshold_type

    def __call__(self, img_):
        if self._threshold_type == "mean":
            mean_pixel = np.mean(img_)
            canny_img = cv2.Canny(img_, 0.6 * mean_pixel, 1.33 * mean_pixel)
        elif self._threshold_type == "median":
            median_pixel = np.median(img_)
            canny_img = cv2.Canny(img_, 0.6 * median_pixel, 1.33 * median_pixel)
        else:
            return None
        return canny_img


class NormalizeImage255:
    def __init__(self):
        pass

    def __call__(self, img_):
        return (img_/255.0).astype('float32')


class NormalizeImage255Canny:
    def __init__(self):
        pass

    def __call__(self, img_):
        return img_/255.0


class NormalizeImage:
    def __init__(self):
        pass

    def __call__(self, img_):
        if np.max(img_) != 0:
            img_ = img_ - np.min(img_)
            img_ = img_ / np.max(img_)
        return img_


class Resize:
    def __init__(self, intended_size):
        self._intended_size = intended_size

    def __call__(self, data):
        return cv2.resize(data, tuple(self._intended_size))


class CustomNormalize:
    def __init__(self, substract_factor, divide_factor):
        self.divide_factor = divide_factor
        self.substract_factor = substract_factor

    def __call__(self, input_array):
        return ((input_array - self.substract_factor) / self.divide_factor).astype('float32')


class CustomDenormalize:
    def __init__(self, add_factor, multiply_factor):
        self.multiply_factor = multiply_factor
        self.add_factor = add_factor

    def __call__(self, input_array):
        return (input_array * self.multiply_factor + self.add_factor).astype('float32')


class Standardization:
    """
    Standardization of whole input array
    """
    def __init__(self):
        pass

    def __call__(self, input_array):
        mean = np.mean(input_array, axis=(0, 1, 2), keepdims=True)
        std = np.std(input_array, axis=(0, 1, 2), keepdims=True)

        return ((input_array - mean) / std).astype('float32')


class GaussKernel:
    """
    Standardization of whole input array
    """
    def __init__(self, kernel_size):
        self.kernel_size = tuple(kernel_size)

    def __call__(self, input_array):
        for i in range(input_array.shape[0]):
            input_array[i] = cv2.GaussianBlur(input_array[i], self.kernel_size, 0)

        return input_array.astype('float32')


if __name__ == "__main__":
    # Executable code intended to conversions testing (should be deleted in final version)
    img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/100000.jpg"
    img = cv2.imread(img_path)

    normalize_image255_canny = NormalizeImage255Canny()
    normalize_image255 = NormalizeImage255()
    # rgb_to_gray = RgbtoGray()
    # filter_image_sobelx = FilterImageSobely()
    # sepia = Sepia()
    # canny = FilterCanny()
    sharpen = FilterSharpen()
    normalize_image = NormalizeImage()
    resize = Resize([512, 512])

    img = resize(img)
    # img = normalize_image255_canny(img)
    # img = sepia(img).astype('float32')
    # gray_img = ImagesConverter.normalize_image255(gray_img)
    img = sharpen(img)
    img = normalize_image255(img)
    # img = np.array(img, dtype='float32')

    print(np.shape(img))
    cv2.imshow(f'filtered_img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/100000.jpg"
    # resize = Resize([512, 512])
    # image = cv2.imread(img_path)
    # kernel = np.array([[-1,-1,-1],
    #                    [-1, 9,-1],
    #                    [-1,-1,-1]])
    # image = resize(image)
    # sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
    # cv2.imshow('Image Sharpening', sharpened)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
