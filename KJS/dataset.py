import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import wandb

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter,RandomRotation,RandomCrop, RandomHorizontalFlip, RandomErasing
from autocrop import Cropper

import matplotlib.image as img
import cv2
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            # RandomRotation(5),
            ToTensor(),
            Normalize(mean=mean, std=std),
            # transforms.Resize(image_size),
        ])

    def __call__(self, image):
        return self.transform(image)

class DatasetFromSubset_Train(Dataset):
    # train data에 먹여 줄 Data Transform
    def __init__(self, subset):
        self.subset = subset
        self.transform = Compose([
            # Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            # RandomRotation(5),
            # ToTensor(),
            # RandomErasing(),
            # Normalize(mean=mean, std=std),
        ])
    def __getitem__(self, index):
        x, y = self.subset[index]
        x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

class CustomAugmentation_for_subset:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            RandomRotation(30),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


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


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None,".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        # vanila_image = plt.imread(image_path)
        # # 테스트#########################################
        # # 얼굴 크롭 되면 return 해주고 아님 말고식
        #
        # # # 1. 얼굴 크롭
        # # # 얼굴크롭이 되는 경우
        # # c = Cropper(face_percent=50)
        # # cropped_img_array = c.crop(image_path)
        # # # 얼굴크롭이 안되는 경우
        # # if str(cropped_img_array) == 'None':
        # #     non_cropped_img = Image.open(image_path)
        # #     # 가로시작점, 세로시작점, 가로범위, 세로범위
        # #     area = (25, 25, 375, 375)
        # #     cropped_img = non_cropped_img.crop(area)
        # #     cropped_img_array = np.array(cropped_img, dtype='uint8')
        #
        # # non_cropped_img = Image.open(image_path)
        # # # 가로시작점, 세로시작점, 가로범위, 세로범위
        # # area = (25, 25, 375, 375)
        # # cropped_img = non_cropped_img.crop(area)
        # # cropped_img_array = np.array(cropped_img, dtype='uint8')
        #
        # # # 2. 선명도 up!!
        # # kernel = np.array([[0, -1, 0],
        # #                   [-1, 5, -1],
        # #                   [0, -1, 0]])
        # # try :
        # #     image_sharp = cv2.filter2D(cropped_img_array, -1, kernel)
        # # except :
        # #     image_sharp = cropped_img_array
        #
        #
        # # # 3. 배경을 지워주자!!
        # # # 마스크 토대가 되는 배열 생성
        # # mask = np.zeros(image_sharp.shape[:2], np.uint8)
        # #
        # # # 내부 알고리즘에 사용하는 사이즈가 (1,65)의np.float64형 배열 생성
        # # bgdModel = np.zeros((1, 65), np.float64)
        # # fgdModel = np.zeros((1, 65), np.float64)
        # #
        # # # 전경 이미지를 감싸는 단형영역을 지정해서 GrabCut로 전경 이미지 추출
        # # rect = (00, 0, 512, 384)
        # #
        # # cv2.grabCut(image_sharp, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        # #
        # # # 업데이트된 마스크를 이용해서 최종적인 마스크 생성
        # # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # # # 입력 이미지와 합성 후 표시
        # # sharp_processed_img = image_sharp * mask2[:, :, np.newaxis]
        # # # opencv에서 PIL로 파일 변경
        # # sharp_processed_img = Image.fromarray(sharp_processed_img)
        # #
        # # # 일단 되는 것만 담고 아님 말고식의 진행이 필요함!!
        # # # 아무것도 전달이 안되었을 때 에러 발생하는 거임
        # #
        # # # 차선책으로 만약 얼굴 인식 못했을 때 중앙을 기준으로 자르는건 어떨까?
        # # # 얼굴 인식을 못하는 사진이 너무 많다...ㅜㅜ
        # # return sharp_processed_img
        #
        # # return Image.fromarray(image_sharp)
        #
        # # return Image.fromarray(cropped_img_array)
        #
        # # return Image.open(image_path)
        #
        # # 2. sharp & background processed
        # # 커널 생성(대상이 있는 픽셀을 강조)
        # kernel = np.array([[0, -1, 0],
        #                    [-1, 5, -1],
        #                    [0, -1, 0]])
        # # 커널 적용
        # image_sharp = cv2.filter2D(vanila_image, -1, kernel)
        # # 이미지 표시용
        # # sharp_image = Image.fromarray(image_sharp)
        #
        # # 선명하게 한 것에 배경을 삭제한다.
        # # 마스크 토대가 되는 배열 생성
        # mask = np.zeros(image_sharp.shape[:2], np.uint8)
        # # 내부 알고리즘에 사용하는 사이즈가 (1,65)의np.float64형 배열 생성
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        # # 가로시작점, 세로시작점, 가로범위, 세로범위
        # rect = (5, 5, 380, 500)
        #
        # cv2.grabCut((image_sharp * 255).astype(np.uint8), mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        # # 업데이트된 마스크를 이용해서 최종적인 마스크 생성
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # # 입력 이미지와 합성 후 표시
        # processed_img = (image_sharp * 255).astype(np.uint8) * mask2[:, :, np.newaxis]
        # image = Image.fromarray(processed_img * 255)
        #
        # print(f'성공 {index}')

        image = Image.open(image_path)
        ##########################################################################################
        return image

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

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        # image = Image.open(self.img_paths[index])
        image_path = self.img_paths[index]
        image = Image.open(self.img_paths[index])
        # #테스트#########################################
        # # 얼굴 크롭 되면 return 해주고 아님 말고식
        #
        # # # 1. 얼굴 크롭
        # # # 얼굴크롭이 되는 경우
        # # c = Cropper(face_percent=50)
        # # cropped_img_array = c.crop(image_path)
        # # # 얼굴크롭이 안되는 경우
        # # if str(cropped_img_array) == 'None' :
        # #     non_cropped_img = Image.open(image_path)
        # #     # 가로시작점, 세로시작점, 가로범위, 세로범위
        # #     area = (25, 25, 375, 375)
        # #     cropped_img = non_cropped_img.crop(area)
        # #     cropped_img_array = np.array(cropped_img, dtype='uint8')
        #
        # # non_cropped_img = Image.open(image_path)
        # # # 가로시작점, 세로시작점, 가로범위, 세로범위
        # # area = (25, 25, 375, 375)
        # # cropped_img = non_cropped_img.crop(area)
        # # cropped_img_array = np.array(cropped_img, dtype='uint8')
        #
        # # # 2. 선명도 up!!
        # # kernel = np.array([[0, -1, 0],
        # #                     [-1, 5, -1],
        # #                     [0, -1, 0]])
        # # try :
        # #     image_sharp = cv2.filter2D(cropped_img_array, -1, kernel)
        # # except :
        # #     image_sharp = cropped_img_array
        #
        # # # 3. 배경을 지워주자!!
        # # # 마스크 토대가 되는 배열 생성
        # # mask = np.zeros(image_sharp.shape[:2], np.uint8)
        # # # 내부 알고리즘에 사용하는 사이즈가 (1,65)의np.float64형 배열 생성
        # # bgdModel = np.zeros((1, 65), np.float64)
        # # fgdModel = np.zeros((1, 65), np.float64)
        # # # 전경 이미지를 감싸는 단형영역을 지정해서 GrabCut로 전경 이미지 추출
        # # rect = (0,0,384,512)
        # # cv2.grabCut(image_sharp, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        # # # 업데이트된 마스크를 이용해서 최종적인 마스크 생성
        # # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # # # 입력 이미지와 합성 후 표시
        # # sharp_processed_img = image_sharp * mask2[:, :, np.newaxis]
        # # # opencv에서 PIL로 파일 변경
        # # sharp_processed_img = Image.fromarray(sharp_processed_img)
        # #
        # # # 일단 되는 것만 담고 아님 말고식의 진행이 필요함!!
        # # # 아무것도 전달이 안되었을 때 에러 발생하는 거임
        # #
        # # # 차선책으로 만약 얼굴 인식 못했을 때 중앙을 기준으로 자르는건 어떨까?
        # # # 얼굴 인식을 못하는 사진이 너무 많다...ㅜㅜ
        # # image = sharp_processed_img
        # # # return Image.open(image_path)
        #
        #
        # # image = Image.fromarray(image_sharp)
        # # 2. sharp & background processed
        # vanila_image = plt.imread(image_path)
        # # 커널 생성(대상이 있는 픽셀을 강조)
        # kernel = np.array([[0, -1, 0],
        #                    [-1, 5, -1],
        #                    [0, -1, 0]])
        # # 커널 적용
        # image_sharp = cv2.filter2D(vanila_image, -1, kernel)
        # # 이미지 표시용
        # sharp_image = Image.fromarray(image_sharp)
        #
        # # 선명하게 한 것에 배경을 삭제한다.
        # # 마스크 토대가 되는 배열 생성
        # mask = np.zeros(image_sharp.shape[:2], np.uint8)
        # # 내부 알고리즘에 사용하는 사이즈가 (1,65)의np.float64형 배열 생성
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        # # 가로시작점, 세로시작점, 가로범위, 세로범위
        # rect = (5, 5, 380, 500)
        #
        # cv2.grabCut((image_sharp * 255).astype(np.uint8), mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        # # 업데이트된 마스크를 이용해서 최종적인 마스크 생성
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # # 입력 이미지와 합성 후 표시
        # processed_img = (image_sharp * 255).astype(np.uint8) * mask2[:, :, np.newaxis]
        # image = Image.fromarray(processed_img * 255)
        #
        # print(f'성공 {index}')
        #
        # ##########################################################################################
        # # return image
        # #
        # # image = Image.fromarray(cropped_img_array)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
