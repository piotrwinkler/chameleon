import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as log
from skimage import color
from torch.utils.data import DataLoader

import base_classes.conversions as conversions
from base_classes.data_collector import DataCollector


class BaseTester:
    """Base class for all tests. When you create your own testing class you should inherit from this one, because it
    contains all parameters from "test_parameters.json" (Provided through entrypoint by SetupCreator) """
    def __init__(self, load_net_path, model, transforms, input_conversions_list, output_conversions_list,
                 dataset_directory, additional_params=dict):
        self._dataset_directory = dataset_directory
        self._load_net_path = load_net_path
        self._model = model
        self._transforms = transforms
        self._input_conversions_list = input_conversions_list
        self._output_conversions_list = output_conversions_list
        self._additional_params = additional_params

        self._files_list = DataCollector.collect_images(self._dataset_directory)

    def __len__(self):
        return len(self._files_list)

    @staticmethod
    def read_image(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def show_images_opencv(images, titles):
        """Beware that opencv requires BGR colors format!"""
        assert len(images) == len(titles), 'Every image should have unique title!'
        for img, title in zip(images, titles):
            cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_images_pyplot(images, titles, cols=1):
        """Beware that pyplot requires RGB colors format!"""
        assert len(images) == len(titles), 'Every image should have unique title!'
        n_images = len(images)
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.axis('off')
            plt.imshow(image)
            a.set_title(title)
        plt.show()

    @staticmethod
    def _implement_transforms(data, transforms):
        for transform in transforms:
            data = transform(data)
        return data

    @staticmethod
    def _implement_conversions(data, conversions_list):
        for conversion in conversions_list:
            data = conversion(data)
        return data


class TestImgtoImg(BaseTester):
    """Class intended to perform tests of img to img networks."""
    def __init__(self, load_net_path, model, transforms, input_conversions_list, output_conversions_list,
                 dataset_directory, additional_params):
        super().__init__(load_net_path, model, transforms, input_conversions_list, output_conversions_list,
                         dataset_directory, additional_params)

    def test(self):
        self._model.load_state_dict(torch.load(self._load_net_path))
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                log.info(f'Name: {name}, Tensor: {param.data}')

        for i in range(len(self)):
            input_img = self.read_image(self._files_list[i])
            input_img = self._implement_conversions(input_img, self._input_conversions_list)
            orig_img = input_img.copy()
            input_img = self._implement_transforms([input_img], self._transforms).pop()
            input_img = input_img.unsqueeze(0)

            output_img = self._model(input_img)
            log.info(f'NN output: {output_img}')
            output_img = output_img.detach().numpy()
            output_img = np.squeeze(output_img)  # remove redundant dimensions
            output_img = output_img.transpose((1, 2, 0)) if len(np.shape(output_img)) == 3 else output_img
            output_img = self._implement_conversions(output_img, self._output_conversions_list)

            log.info(f'Original image shape: {np.shape(orig_img)}')
            log.info(f'Output image shape: {np.shape(output_img)}')

            self.show_images_pyplot([orig_img, output_img], ['original', 'NN'])


class ImageColorizationTester(BaseTester):
    def __init__(self, load_net_path, network, dataset, results_dir, config_dict):
        super().__init__(load_net_path, network, [], [], [], '.', additional_params=config_dict['additional_params'])
        self.dataset = dataset
        self.dataloader_params = config_dict['dataloader_parameters']
        self.results_dir = results_dir

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else
                                    'cpu')
        log.info(self._device)
        self._test_on_gpu = config_dict['test_on_gpu']
        log.debug(f"Choosing net {config_dict['net']}")

    def test(self):

        dataloader = DataLoader(self.dataset, **self.dataloader_params)
        self._model.load_state_dict(torch.load(self._load_net_path))
        self._model.eval()
        if self._test_on_gpu:
            self._model = self._model.to(self._device)

        with torch.no_grad():
            for i, (L_batch_gray_not_processed, rgb_images, L_batch_gray, gray_images) in enumerate(dataloader):
                L_gray_not_processed = np.transpose(L_batch_gray_not_processed[0].numpy(), (1, 2, 0))

                fig = plt.figure(figsize=(16, 8))
                ax1 = fig.add_subplot(1, 4, 1)
                ax1.imshow(rgb_images[0])
                ax1.title.set_text('Ground Truth')

                ax2 = fig.add_subplot(1, 4, 2)
                ax2.imshow(gray_images[0])
                ax2.title.set_text('Gray')

                ax3 = fig.add_subplot(1, 4, 3)
                ax3.imshow(np.transpose(L_batch_gray[0].numpy(), (1, 2, 0)).squeeze())
                ax3.title.set_text(f"gray L channel, blur={self._additional_params['blur']['do_blur']}")

                if self._test_on_gpu:
                    L_batch_gray = L_batch_gray.to(self._device)

                outputs = self._model(L_batch_gray)

                if self._test_on_gpu:
                    outputs = outputs.cpu()

                ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
                if self._additional_params['ab_output_processing'] == "normalization":
                    ab_outputs = ab_outputs * 255

                elif self._additional_params['ab_output_processing'] == "standardization":
                    ab_outputs = ab_outputs * self.dataset.ab_std[0] + self.dataset.ab_mean[0]

                elif self._additional_params['ab_output_processing'] == "trick":
                    scale_L = L_gray_not_processed / 100
                    scale = max([np.max(ab_outputs), abs(np.min(ab_outputs))])
                    ab_outputs = ab_outputs / scale
                    ab_outputs = ab_outputs * (scale_L * 127)

                img_rgb_outputs = color.lab2rgb(np.dstack((L_gray_not_processed, ab_outputs)))

                ax4 = fig.add_subplot(1, 4, 4)
                ax4.imshow(img_rgb_outputs)
                ax4.title.set_text('model output')
                if self._additional_params['do_show_results']:
                    plt.show()

                if self._additional_params['do_save_results']:
                    matplotlib.image.imsave(f"{self.results_dir}/{str(i).zfill(4)}.png", img_rgb_outputs)

                if i == self._additional_params['how_many_results_to_generate']:
                    break
                plt.close(fig)

        log.info('Finished Testing')

    def __len__(self):
        return len(self.dataset)
