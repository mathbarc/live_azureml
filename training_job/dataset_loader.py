import os
import random
from enum import Enum
from typing import List

import logging

from torchvision.transforms import Compose
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2

from azureml.fsspec import AzureMachineLearningFileSystem

from config import ml_client

class DatasetType(Enum):
    TRAIN = 0
    TEST = 1

class ImageClassificationDataset(Dataset):

    CACHE_PATH="/tmp"

    def __init__(self, dataset_name:str, version:int, dataset_type:DatasetType = DatasetType.TRAIN, transform:Compose=None, target_transform:Compose=None, labels:List[str]=None):
        logging.info("ImageClassificationDataset -> Connecting to dataset filesystem")
        filedataset_asset = ml_client.data.get(name=dataset_name, version=str(version))
        self.azure_fs = AzureMachineLearningFileSystem(filedataset_asset.path)

        folders = self.azure_fs.ls()
        self.azure_data_root = f"{os.sep}".join(folders[0].split(os.sep)[:-2])

        self.transform = transform
        self.target_transform = target_transform
        logging.info("ImageClassificationDataset -> Connecting to dataset filesystem ... done")
        logging.info(f"ImageClassificationDataset -> loading dataset {dataset_name}:{version} metadata")
        self._load_dataset_from_azureml(dataset_type)
        if labels is None:
            self.labels = list(set([label for _, label in self.image_list]))
        else:
            self.labels = labels
        logging.info(f"ImageClassificationDataset -> loading dataset {dataset_name}:{version} metadata ... done")

    def _load_dataset_from_azureml(self, dataset_type:DatasetType = DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            path = os.path.join(self.azure_data_root, "train")
        elif dataset_type == DatasetType.TEST:
            path = os.path.join(self.azure_data_root, "test")

        self.image_list = []

        for file in self.azure_fs.glob(f"{path}/*/*.jpg"):
            label = file.split(os.sep)[-2]
            self.image_list.append((file,label))

        

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        img_path, label_str = self.image_list[index]
        local_img_path = os.path.join(ImageClassificationDataset.CACHE_PATH, img_path)
        if not os.path.exists(local_img_path):
            self.azure_fs.get_file(img_path, ImageClassificationDataset.CACHE_PATH)

        image = cv2.imread(local_img_path)
        label = self.labels.index(label_str)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
            


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    dataset = ImageClassificationDataset("rice_dataset", 6)

    img, label = dataset[-1]

    print(len(dataset))
    print(img.shape)
    print(label)
    cv2.imshow("img", img)
    cv2.waitKey()
