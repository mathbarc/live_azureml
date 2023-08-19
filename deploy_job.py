from config import ml_client
from azureml.fsspec import AzureMachineLearningFileSystem
import os

filedataset_asset = ml_client.data.get(name="rice_dataset", version="3")

fs = AzureMachineLearningFileSystem(filedataset_asset.path)

folders = fs.ls()
root = f"{os.sep}".join(folders[0].split(os.sep)[:-2])


train_files = []
test_files = []


for file in fs.glob(f"{root}/train/*/*.jpg"):
    train_files.append(file)

for file in fs.glob(f"{root}/test/*/*.jpg"):
    test_files.append(file)



...
