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


class RestrictValues:
    def __init__(self):
        pass

    def __call__(self, input_array):
        new = input_array + abs(np.min(input_array))
        return new/np.max(new)


if __name__ == "__main__":
    #   Executable code intended to conversions testing (should be deleted in final version)
    img_path5 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/124003.jpg"     # stateczek
    img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/126803.jpg"      # skały
    img_path2 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/100000.jpg"
    img_path3 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/101801.jpg"
    img_path4 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/112602.jpg"

    # img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    img3 = cv2.imread(img_path3)
    img4 = cv2.imread(img_path4)
    img5 = cv2.imread(img_path5)

    normalize_image255_canny = NormalizeImage255Canny()
    normalize_image255 = NormalizeImage255()
    rgb_to_gray = RgbtoGray()
    filter_image_sobelx = FilterImageSobelx()
    sepia = Sepia()
    canny = FilterCanny()
    sharpen = FilterSharpen()
    normalize_image = NormalizeImage()
    resize = Resize([256, 256])

    img = resize(img5)
    img2 = resize(img2)
    img3 = resize(img3)
    img4 = resize(img4)
    # img = normalize_image255_canny(img)
    # img = sepia(img).astype('float32')
    # gray_img = ImagesConverter.normalize_image255(gray_img)
    img = normalize_image255(img)
    img = rgb_to_gray(img)

    img = filter_image_sobelx(img)
    # img = sepia(img)
    # img = sharpen(img)

    # img = np.array(img, dtype='float32')
    img = normalize_image(img)
    print(np.shape(img))
    cv2.imshow(f'img', img)
    # cv2.imshow(f'img2', img2)
    # cv2.imshow(f'img3', img3)
    # cv2.imshow(f'img4', img4)
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

# def show_images(images, cols=2, titles=None):
#     """Display a list of images in a single figure with matplotlib.
#
#     Parameters
#     ---------
#     images: List of np.arrays compatible with plt.imshow.
#
#     cols (Default = 1): Number of columns in figure (number of rows is
#                         set to np.ceil(n_images/float(cols))).
#
#     titles: List of titles corresponding to each image. Must have
#             the same length as titles.
#     """
#     assert ((titles is None) or (len(images) == len(titles)))
#     n_images = len(images)
#     if titles is None:
#         titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
#     fig = plt.figure()
#     for n, (image, title) in enumerate(zip(images, titles)):
#         a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
#         if image.ndim == 2:
#             plt.gray()
#         plt.axis('off')
#         plt.imshow(image)
#         # a.set_title(title)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#     plt.show()
#

# import matplotlib.pyplot as plt
# import numpy as np
#
# img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/124003.jpg"
# img_path2 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/118200.jpg"
# img_path3 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/101801.jpg"
# img_path4 = "/home/piotr/venvs/inz/projects/chameleon/datasets/training_dataset/112602.jpg"
# list_ = [plt.imread(img_path), plt.imread(img_path2), plt.imread(img_path3), plt.imread(img_path4)]
# show_images(list_)
