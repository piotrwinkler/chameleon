"""Setup creator use this script to get all conversions mentioned in "training_parameters.json" or in
"test_parameters.json" """
import cv2
import numpy as np


class RgbtoGray:
    def __init__(self):
        pass

    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')


class FilterImageSobelx:
    def __init__(self):
        pass

    def __call__(self, img):
        return cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3).astype('float32')


class FilterImageSobely:
    def __init__(self):
        pass

    def __call__(self, img):
        return cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3).astype('float32')


class Sepia:
    def __init__(self):
        self._kernel = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]])

    def __call__(self, img):
        img_sepia = cv2.transform(img, self._kernel)
        # Check which entries have a value greater than 255 and set it to 255
        img_sepia[np.where(img_sepia > 255)] = 255
        return img_sepia


class NormalizeImage255:
    def __init__(self):
        pass

    def __call__(self, img):
        return (img/255.0).astype('float32')


class NormalizeImage:
    def __init__(self):
        pass

    def __call__(self, img):
        if np.max(img) != 0:
            img = img - np.min(img)
            img = img / np.max(img)
        return img


class Resize:
    def __init__(self, intended_size):
        self._intended_size = intended_size

    def __call__(self, data):
        return cv2.resize(data, tuple(self._intended_size))


if __name__ == "__main__":
    # Executable code intended to conversions testing (should be deleted in final version)
    img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/test_dataset/samolot.jpg"
    img = cv2.imread(img_path).astype(np.float32)

    normalize_image255 = NormalizeImage255()
    # rgb_to_gray = RgbtoGray()
    # filter_image_sobelx = FilterImageSobely()
    sepia = Sepia()
    normalize_image = NormalizeImage()
    resize = Resize([512, 512])

    img = resize(img)
    img = normalize_image255(img)
    # img = sepia(img).astype('float32')
    # gray_img = ImagesConverter.normalize_image255(gray_img)
    img = sepia(img).astype('float32')
    img = np.array(img, dtype='float32')

    cv2.imshow(f'filtered_img', normalize_image(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
