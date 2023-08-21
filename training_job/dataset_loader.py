import os
from enum import Enum
from typing import List

import logging

from torchvision.io.image import read_image
from torchvision.transforms import Compose
from torch.utils.data import Dataset


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1

class ImageClassificationDataset(Dataset):

    CACHE_PATH="/tmp"

    def __init__(self, path, dataset_type:DatasetType = DatasetType.TRAIN, transform:Compose=None, target_transform:Compose=None, labels:List[str]=None):
        
        if dataset_type == DatasetType.TRAIN:
            folder_path = os.path.join(path, "train")
        elif dataset_type == DatasetType.TEST:
            folder_path = os.path.join(path, "train")
        else:
            raise Exception("Invalid Dataset Type")

        self.transform = transform
        self.target_transform = target_transform

        self._load_dataset(folder_path)
        if labels is None:
            self.labels = list(set([label for _, label in self.image_list]))
        else:
            self.labels = labels

    def _load_dataset(self, path):
        
        self.image_list = []

        for label in os.listdir(path):
                folder_path = os.path.join(path, label)
                logging.info(f"Loading samples of {label}")
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".jpg"):
                        self.image_list.append((os.path.join(folder_path,file_name),label))

        

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        img_path, label_str = self.image_list[index]
        
        image = read_image(img_path).float() *(1./255.)
        
        label = self.labels.index(label_str)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
            


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    dataset = ImageClassificationDataset("./datasets/Rice_Image_Dataset")

    img, label = dataset[-1]

    print(len(dataset))
    print(img.shape)
    print(label)