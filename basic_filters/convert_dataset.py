import os
import glob
from basic_filters.basic_filters_class import ImagesConverter

from_path = "datasets/inside_city/"
to_path = "datasets/inside_city_converted_sephia/"


def main():
    if not os.path.isdir(from_path):
        raise Exception("Unable to find path with input dataset")

    if not os.path.isdir(to_path):
        os.mkdir(to_path)
        print(f"Created directory at '{to_path}'")

    number_of_images = len(os.listdir(from_path))

    image_converter = ImagesConverter()

    for counter, img_path in enumerate(glob.glob(from_path + "*")):
        print(f"processing image {counter+1} on {number_of_images}")

        try:
            bgr_img = image_converter.load_img(img_path)
            img_sepia_bgr = image_converter.bgr_to_sephia(bgr_img)
            image_converter.save_img(to_path + os.path.basename(img_path), img_sepia_bgr)

        except Exception as e:
            print(f"Error with file: {img_path} - {e}")


if __name__ == "__main__":
    main()
