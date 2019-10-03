import os
import glob
from basic_filters.basic_filters_class import BasicFilters

from_path = "datasets/inside_city/"
to_path = "datasets/inside_city_converted/"


def main():
    if not os.path.isdir(from_path):
        raise Exception("Unable to find path with input dataset")

    if not os.path.isdir(to_path):
        os.mkdir(to_path)

    number_of_images = len(os.listdir(from_path))

    basic_filters = BasicFilters()

    for counter, img_path in enumerate(glob.glob(from_path + "*")):
        print("processing image {} on {}".format(counter+1, number_of_images))

        try:
            img = basic_filters.load_img(img_path)
            gray = basic_filters.rgb_to_gray(img)
            img_sobel_64 = basic_filters.sobel_filter(gray, depth="64F", kernel_size=3)
            basic_filters.save_img(to_path + os.path.basename(img_path), img_sobel_64)

        except Exception as e:
            print("Error with file: {} - {}".format(img_path, e))


if __name__ == "__main__":
    main()
