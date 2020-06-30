import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random


class DatasetShopee(Dataset):
    root_dir = Path("~/data/shopee").expanduser()

    def __init__(self, csv_name, data_dir, transforms=None) -> None:
        self.transforms = transforms
        csv_path = self.root_dir / csv_name
        self.data_dir = self.root_dir / data_dir
        df = pd.read_csv(csv_path)
        df['filename'] = df['filename']
        self.df = df

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        category, filename = row['category'], row['filename']
        if category == 43:
            image_path = f'{self.data_dir.__str__()}/{filename}' 
        else:
            image_path = f'{self.data_dir.__str__()}/{category:02}/{filename}' 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, category, filename

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    train_set = DatasetShopee("train.csv", "train")
    print(len(train_set))
    image, label = train_set[0]
    print(image.shape, label)
    test_set = DatasetShopee("test.csv", "test")
    print(len(test_set))
    image, label = test_set[100]
    print(image.shape, label)
