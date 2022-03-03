import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
from torch import optim,nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

from collections import defaultdict
from typing import Tuple, List
import random

class MaskDataset(Dataset):
    def __init__(self, base_dir, df, transform=None):
        self.base_dir = base_dir
        self.df = df
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label

        X = Image.open(self.df.iloc[index]['path'])
        y = torch.tensor(int(self.df.iloc[index]['targets']))
        if self.transform:

            X = self.transform(X)

        return X, y

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

from enum import Enum
from torch.utils.data import Dataset
import random


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD

class Subset(object):
    def __init__(self, image_path, indices, transform):
        self.image_path = image_path
        self.indices = indices
        self.transform = transform

        self.image = []
        for i in range(len(self.image_path)):
            self.image.append(self.read_image(i))

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.indices[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)

    def read_image(self, index):
        image_path = self.image_path[index]
        """
        준석이 코드
        """
        return Image.open(image_path)


class CustomMaskSplitByProfileDataset(Dataset):
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = {'train':[], 'val':[]}
    # mask_labels = {'train':[], 'val':[]}
    # gender_labels = {'train':[], 'val':[]}
    # age_labels = {'train':[], 'val':[]}
    multi_class_label = {'train':[], 'val':[]}

    def __init__(self, data_dir, val_ratio=0.2):

        self.data_dir = data_dir
        #         self.setup_basic()
        self.val_ratio = val_ratio

        self.setup()

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        # val_indices = set(random.choices(range(length), k=5))

        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }


    def set_transform(self, train_transform, val_transform):
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, shuffle=False):
        profiles = os.listdir(self.data_dir)

        if shuffle:
            import random
            random.shuffle(profiles)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)


        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile,
                                            file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths[phase].append(img_path)
                    # self.mask_labels[phase].append(mask_label)
                    # self.gender_labels[phase].append(gender_label)
                    # self.age_labels[phase].append(age_label)

                    label = self.encode_multi_class(mask_label, gender_label, age_label)
                    self.multi_class_label[phase].append(label)


    def split_dataset(self) -> List[Subset]:

        train_sub = Subset(self.image_paths['train'], self.multi_class_label['train'], self.train_transform)
        val_sub = Subset(self.image_paths['val'], self.multi_class_label['val'], self.val_transform)
        return train_sub, val_sub


def preprocess_df(base_dir, mode='train'):

    ### check for directory, file, extension
    train_data_dir = glob(os.path.join(base_dir, mode, 'images', '*'))
    train_data_path = glob(os.path.join(base_dir, mode, 'images', '*', '*'))
    print('num of directory', len(train_data_dir))
    print('num of file', len(train_data_path))
    num_ext = defaultdict(int)

    for file in train_data_path:
        ext = file.split('.')[-1]
        if ext not in num_ext.keys():
            num_ext[ext] += 1
        else:
            num_ext[ext] += 1
    print('num of file extension', num_ext)

    assert len(train_data_path) == len(train_data_dir)*7
    assert len(train_data_path) == sum(num_ext.values())


    ### check null (mislabelling 제외 > 토론게시판 참고)
    df_train = pd.read_csv(os.path.join(base_dir, 'train', 'train.csv'))
    print(df_train.isnull().sum())

    ### check range
    print('min value', df_train['age'].min())
    print('max value', df_train['age'].max())

    ### new data frame including all the images
    new_dict = {'id': [],
                'gender': [],
                'race': [],
                'age': [],
                'mask': [],
                'path': [],
                'dir': []}
    """
    큰일날뻔;
    """
    data_path = '/opt/ml/input/data'
    for idx, df_train_data in enumerate(df_train.iterrows()):
        data_dir = glob(os.path.join(data_path, 'train', 'images', df_train_data[1].path, '*'))
        for data_file in data_dir:
            if 'incorrect' in os.path.basename(data_file):
                new_dict['mask'].append('incorrect')
            elif 'normal' in os.path.basename(data_file):
                new_dict['mask'].append('not_wear')
            else:
                new_dict['mask'].append('wear')
            new_dict['id'].append(df_train_data[1].id)
            new_dict['gender'].append(df_train_data[1].gender)
            new_dict['race'].append(df_train_data[1].race)
            new_dict['age'].append(df_train_data[1].age)
            new_dict['path'].append(data_file)
            new_dict['dir'].append(df_train_data[1].path)
    new_train_df = pd.DataFrame(new_dict)

    filters = [
        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 60),

        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 60),

        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'male') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 60),

        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'female') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'incorrect') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 60),

        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'male') & (new_train_df['age'] >= 60),

        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] < 30),
        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 30) & (
                    new_train_df['age'] < 60),
        (new_train_df['mask'] == 'not_wear') & (new_train_df['gender'] == 'female') & (new_train_df['age'] >= 60),
    ]

    values = list(range(18))
    new_train_df['targets'] = np.select(filters, values)

    ### class imbalance check
    print('final class distributions')
    print(new_train_df['targets'].value_counts())
    new_train_df_info = dict(new_train_df['targets'].value_counts())
    return new_train_df, new_train_df_info


if __name__=="__main__":
    base_dir = '/opt/ml/input/data'
    train_df = preprocess_df(base_dir)
    # train_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomVerticalFlip(),
    #                                       transforms.RandomRotation(20),
    #                                       transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(norm_mean, norm_std)])
    input_size = 256
    train_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    train_dataset = MaskDataset(base_dir, train_df, transform = train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    device = 'cuda'
    model_ft = models.resnet18(pretrained=True)
    num_classes = 18
    num_ftrs = 512

    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.to(device)

    from sklearn.model_selection import train_test_split
    train, valid = train_test_split(train_df,test_size=0.2,
                                    shuffle=True, stratify=train_df['targets'])

    train_dataset = MaskDataset(base_dir, train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    val_dataset = MaskDataset(base_dir, valid, transform=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        breakpoint()

