from training_job.config import ml_client
from azureml.fsspec import AzureMachineLearningFileSystem
import os
import cv2

filedataset_asset = ml_client.data.get(name="rice_dataset", version="6")

fs = AzureMachineLearningFileSystem(filedataset_asset.path)

folders = fs.ls()
root = f"{os.sep}".join(folders[0].split(os.sep)[:-2])


train_files = []
test_files = []


for file in fs.glob(f"{root}/train/*/*.jpg"):
    train_files.append(file)


img_path = os.path.join("/tmp",train_files[0])

if not os.path.exists(img_path):
    fs.get_file(train_files[0],"/tmp")

img = cv2.imread(img_path)

cv2.imshow("tmp", img)
cv2.waitKey()


...
