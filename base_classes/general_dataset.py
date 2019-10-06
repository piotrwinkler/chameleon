import glob
import torch
from torch.utils.data import Dataset, DataLoader


class GeneralDataset(Dataset):
    def __init__(self, dataset_directory):
        self.file_types = [f'{dataset_directory}.jpg', f'{dataset_directory}.jpeg',
                           f'{dataset_directory}.png', f'{dataset_directory}.bmp']
        self.files_list = [glob.glob(e) for e in self.file_types]

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
