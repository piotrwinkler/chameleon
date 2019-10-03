import numpy as np
import cv2
from basic_filters.basic_filters_class import BasicFilters

img_path = 'datasets/street/art379.jpg'
# img_path = 'datasets/samolot_mini_mini.jpg'


def main():

    basic_filters = BasicFilters()

    img = cv2.imread(img_path)
    basic_filters.show_image_without_waitkey(img, title="Original Image")

    gray = basic_filters.rgb_to_gray(img)
    basic_filters.show_image_without_waitkey(gray, title="Gray")

    # Canny (mean is the best)
    img_canny_mean = basic_filters.canny_filter(img, threshold_type="mean")
    basic_filters.show_image_without_waitkey(img_canny_mean, title="Canny mean")

    img_canny_median = basic_filters.canny_filter(img, threshold_type="median")
    basic_filters.show_image_without_waitkey(img_canny_median, title="Canny median")

    img_canny_median = basic_filters.canny_filter(img, threshold_type="manual")
    basic_filters.show_image_without_waitkey(img_canny_median, title="Canny hardcoded")

    # Sobel
    img_sobel = basic_filters.sobel_filter(gray, depth="8U", kernel_size=3)
    basic_filters.show_image_without_waitkey(img_sobel, title="Sobel 8U")

    img_sobel_64 = basic_filters.sobel_filter(gray, depth="64F", kernel_size=3)
    basic_filters.show_image_without_waitkey(img_sobel_64, title="Sobel 64F")

    # Prewitt
    img_prewitt = basic_filters.prewitt_filter(gray)
    basic_filters.show_image_without_waitkey(img_prewitt, title="Prewitt")

    # Roberts cross
    img_roberts_cross = basic_filters.roberts_cross_filter(gray)
    basic_filters.show_image_without_waitkey(img_roberts_cross, title="Roberts Cross")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
