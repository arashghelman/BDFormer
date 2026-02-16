from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
      
class Skin_Dataset(Dataset):
    def __init__(self, config, split='train', subset_frac=None):
        super().__init__()

        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"

        self.split = split
        self.path = os.path.join(config.data_path, split + '/')
        self.data = []

        if split == "train":
            self.transformer = config.train_transformer
        else:
            self.transformer = config.test_transformer

        labels_df = pd.read_csv(self.path + 'labels.csv')
        labels_df['encoded_labels'] = labels_df['lesion_type'].str.lower().map(config.class_map)
        self.labels = labels_df['encoded_labels'].to_numpy()
        labels_map = dict(zip(labels_df['image_id'], labels_df['encoded_labels']))

        for img_id in labels_df['image_id']:
            img_path = os.path.join(self.path, 'images', f"{img_id}.jpg")
            mask_path = os.path.join(self.path, 'masks', f"{img_id}_segmentation.png")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.data.append([img_path, mask_path, labels_map[img_id]])
        
        if subset_frac is not None and subset_frac < 1.0:
            labels_for_strat = [item[2] for item in self.data]
            
            self.data, _, _, _ = train_test_split(
                self.data, 
                labels_for_strat, 
                train_size=subset_frac, 
                stratify=labels_for_strat, 
                random_state=42
            )
            
            print(f"Subsetting {split} set to {len(self.data)} samples.")

    def __getitem__(self, indx):
        img_path, msk_path, label = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        contour = np.expand_dims(cv2.Canny(msk.astype(np.uint8), 0, 1), axis=2) / 255
        contour = cv2.resize(contour, (256, 256))
        kernel = np.ones((9, 9), np.uint8)
        contour = cv2.dilate(contour, kernel).reshape([256, 256, 1])
        img, msk, contour = self.transformer((img, msk, contour))

        return img, msk, contour, label

    def __len__(self):
        return len(self.data)

from config_setting import setting_config_multitask as config

def main():
    dataset = Skin_Dataset(config, 'train', 0.2)
    print(dataset.labels)
    
    # for i, row in enumerate(dataset.data):
    #     img_path, mask_path, label = row
    #     print(f"Row {i}:")
    #     print(f"  Image: {img_path}")
    #     print(f"  Mask:  {mask_path}")
    #     print(f"  Label: {label}")
    #     print("-" * 30)

if __name__ == '__main__':
    main()