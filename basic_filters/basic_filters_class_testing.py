import numpy as np
import cv2
from basic_filters.basic_filters_class import ImagesConverter

# img_path = 'datasets/red.png'
img_path = 'datasets/lena_color.jpg'


def main():

    image_converter = ImagesConverter()

    bgr_img = cv2.imread(img_path)

    bgr_gausian_blur = image_converter.gaussian_blur(bgr_img)
    image_converter.show_image_without_waitkey(bgr_gausian_blur, title="BGR Gausian blue")

    # rgb_img = image_converter.bgr_to_rgb(img)
    image_converter.show_image_without_waitkey(bgr_img, title="Original Image")

    gray = image_converter.bgr_to_gray(bgr_img)
    image_converter.show_image_without_waitkey(gray, title="Gray")

    # Canny (mean is the best)
    img_canny_mean = image_converter.canny_filter(bgr_img, threshold_type="mean")
    image_converter.show_image_without_waitkey(img_canny_mean, title="Canny mean")

    img_canny_median = image_converter.canny_filter(bgr_img, threshold_type="median")
    image_converter.show_image_without_waitkey(img_canny_median, title="Canny median")

    img_canny_median = image_converter.canny_filter(bgr_img, threshold_type="manual")
    image_converter.show_image_without_waitkey(img_canny_median, title="Canny hardcoded")

    # Sobel
    img_sobel = image_converter.sobel_filter(gray, depth="8U", kernel_size=3)
    image_converter.show_image_without_waitkey(img_sobel, title="Sobel 8U")

    img_sobel_64 = image_converter.sobel_filter(gray, depth="64F", kernel_size=3)
    image_converter.show_image_without_waitkey(img_sobel_64, title="Sobel 64F")

    # Prewitt
    img_prewitt = image_converter.prewitt_filter(gray)
    image_converter.show_image_without_waitkey(img_prewitt, title="Prewitt")

    # Roberts cross
    img_roberts_cross = image_converter.roberts_cross_filter(gray)
    image_converter.show_image_without_waitkey(img_roberts_cross, title="Roberts Cross")

    # Sepia from BGR
    img_sepia_bgr = image_converter.bgr_to_sephia(bgr_img)
    image_converter.show_image_without_waitkey(img_sepia_bgr, title="Sepia bgr")

    # Sepia from RGB
    rgb_img = image_converter.bgr_to_rgb(bgr_img)
    img_sepia_rgb = image_converter.rgb_to_sephia(rgb_img)
    img_sepia_bgr = image_converter.rgb_to_bgr(img_sepia_rgb)
    image_converter.show_image_without_waitkey(img_sepia_bgr, title="Sepia rgb")

    # Negative from BGR
    negative_img = image_converter.negative_filter(bgr_img)
    image_converter.show_image_without_waitkey(negative_img, title="Negative")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
