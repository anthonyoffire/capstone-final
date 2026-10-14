import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class AgeDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.normpath(self.df.iloc[idx]['file_path'])
        age = float(self.df.iloc[idx]['age'])

        age = (age - 5) / (70 - 5)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32), img_path
