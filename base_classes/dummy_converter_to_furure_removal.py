"""Setup creator use this script to get all conversions mentioned in "training_parameters.json" or in
"test_parameters.json" """
import cv2
import numpy as np


class RgbtoGray:
    def __init__(self):
        pass

    def __call__(self, img):
        # print(np.shape(img))
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
    img_path = "/home/piotr/venvs/inz/projects/chameleon/datasets/test_dataset/arab.jpg"
    img = cv2.imread(img_path).astype(np.float32)

    normalize_image255 = NormalizeImage255()
    rgb_to_gray = RgbtoGray()
    filter_image_sobelx = FilterImageSobelx()
    normalize_image = NormalizeImage()
    resize = Resize([256, 256])

    img = resize(img)
    img = normalize_image255(img)
    img = rgb_to_gray(img).astype('float32')
    # gray_img = ImagesConverter.normalize_image255(gray_img)
    img = filter_image_sobelx(img).astype('float32')
    img = np.array(img, dtype='float32')

    cv2.imshow(f'filtered_img', normalize_image(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
