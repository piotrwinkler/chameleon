import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from base_classes.data_collector import DataCollector
from loguru import logger as log
import matplotlib.pyplot as plt


class BaseTester:
    """Base class for all tests. When you create your own testing class you should inherit from this one, because it
    contains all parameters from "test_parameters.json" (Provided through entrypoint by SetupCreator) """
    def __init__(self, net_path, model, transforms, input_conversions_list, output_conversions_list, dataset_directory):
        self._dataset_directory = dataset_directory
        self._net_path = net_path
        self._model = model
        self._transforms = transforms
        self._input_conversions_list = input_conversions_list
        self._output_conversions_list = output_conversions_list

        self._files_list = DataCollector.collect_images(self._dataset_directory)

    def __len__(self):
        return len(self._files_list)

    @staticmethod
    def read_image(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def show_image(imgs_list):
        for i, img in enumerate(imgs_list):
            cv2.imshow(f'image{i}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    def __init__(self, net_path, model, transforms, input_conversions_list, output_conversions_list, dataset_directory):
        super().__init__(net_path, model, transforms, input_conversions_list, output_conversions_list,
                         dataset_directory)

    def test(self):
        self._model.load_state_dict(torch.load(self._net_path))

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

            self.show_image([orig_img, output_img])


class ImageColorizationTester(BaseTester):
    def __init__(self, net_path, model, dataset, dataloader_params, do_save_results, deconversion_details):
        super().__init__(net_path, model, [], [], [], '.')
        self.dataset = dataset
        self.dataloader_params = dataloader_params
        self.do_save_results = do_save_results
        self.deconversion_details = deconversion_details

    def test(self):

        dataloader = DataLoader(self.dataset, **self.dataloader_params)
        self._model.load_state_dict(torch.load(self._net_path))
        self._model.eval()
        log.info(f"Choosing net {str(self._model)}")

        with torch.no_grad():
            for i, (L_batch_gray_no_blurred, ab_batch_rgb, rgb_images, L_batch_gray, gray_images) in enumerate(dataloader):
                L_gray_not_blured = np.transpose(L_batch_gray_no_blurred[0].numpy(), (1, 2, 0))

                fig = plt.figure(figsize=(14, 7))
                ax1 = fig.add_subplot(1, 4, 1)
                ax1.imshow(rgb_images[0])
                ax1.title.set_text('Ground Truth')

                ax2 = fig.add_subplot(1, 4, 2)
                ax2.imshow(gray_images[0])
                ax2.title.set_text('Gray')

                ax3 = fig.add_subplot(1, 4, 3)
                ax3.imshow(np.transpose(L_batch_gray[0].numpy(), (1, 2, 0)).squeeze())
                ax3.title.set_text(f"gray L channel, blur={self.dataset.blur_details['do_blur']}")

                outputs = self._model(L_batch_gray)

                if L_input_processing == "normalization":
                    L_gray = L_gray * 100 + 50

                elif L_input_processing == "standardization":
                    L_gray = L_gray * cifar_dataset.L_std[0] + cifar_dataset.L_mean[0]

                ab_outputs = np.transpose(outputs[0].numpy(), (1, 2, 0))
                if ab_output_processing == "normalization":
                    ab_outputs = ab_outputs * 255

                elif ab_output_processing == "standardization":
                    ab_outputs = ab_outputs * cifar_dataset.ab_std[0] + cifar_dataset.ab_mean[0]

                elif ab_output_processing == "trick":
                    scale_L = L_gray / 100
                    scale = max([np.max(ab_outputs), abs(np.min(ab_outputs))])
                    ab_outputs = ab_outputs / scale
                    ab_outputs = ab_outputs * (scale_L * 127)

                img_rgb_outputs = color.lab2rgb(np.dstack((L_gray, ab_outputs)))

                ax4 = fig.add_subplot(1, 4, 4)
                ax4.imshow(img_rgb_outputs)
                ax4.title.set_text('model output')
                if do_show_results:
                    plt.show()

                if do_save_results:
                    matplotlib.image.imsave(f"{results_dir}/{str(i).zfill(4)}.png", img_rgb_outputs)

                running_loss = loss.item()

                print(f'[{(i + 1) * batch_size}] loss: {running_loss}')

                if i == how_many_results_to_generate:
                    break

        print('Finished Testing')

    def __len__(self):
        return len(self.dataset)
