from base_classes.tester import Tester
from base_classes.transforms import Rescale, Normalize, ToTensor
from canny_filter import CannyFIlter
from data import consts


def main():
    test_config = {
            'image_path': f'{consts.DATASET_DIRECTORY}/arab.jpg',
            'net_weights_path': consts.NET_SAVING_DIRECTORY,
            'net_model': CannyFIlter(),
            'transforms':   [{'type': Rescale,
                            'arguments': [[(256, 256), (256, 256)]]},
                            {'type': Normalize,
                            'arguments': []},
                            {'type': ToTensor,
                            'arguments': []}],
                   }

    img = Tester.read_image(test_config['image_path'])
    test_conf = {'input_img': img, 'net_path': test_config['net_weights_path'], 'model': test_config['net_model'],
                 'transfroms': test_config['transforms']}
    Tester.test_img_img_network(**test_conf)


if __name__ == "__main__":
    main()
