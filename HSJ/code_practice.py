from code.baseline_v2.dataset import IMG_EXTENSIONS


IMG_EXTENSIONS = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.bmp', '.BMP' ]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, image):
        return self.transform(image)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor): #인스턴스가 호출'인스턴스이름()'했을 때 실행됨
        return tensor + torch.randn(tensor.size())*self.std + self.mean
    
    def __repr__(self): #인스턴스를 print할 때 출력됨(str과 유사한 역할, 대신 str가 우선순위가 더 높음)
        return self.__class__.__name__ + f'(mean= {self.mean}, std={self.std})'


class MaskLabels(int, Enum):
    MASK=0
    INCORRECT = 1
    NORMAL = 2

class GenserLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(clas, vlaue: str) -> int: #이함수가 반환해주는 값이 int 형식임을 의미
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female'")

class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_numbers(cls, value)

    #일단 생략


class MaskBaseDataset(Dataset):
    num_classes = 3*2*3

    _file_names = {
        "mask1": MaskLabels.MASk,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK
    }

    image_paths = []
    maske_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transfrom = None
        self.setup()
        self.calc_statistics()
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile


#----------------------------------------------------
import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import Dataloader

from dataset import TestDataset, MaskBaseDataset

def load_model(save_model, num_classes, device)L
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes = num_classes
    )

    model_path  = os.path.join(save_model, 'best.pth')
    model.load_statd_dict(torch.load(model.pth, map_location=device))

    return model

def inference(data_dir, model_dir, output_dir, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    numc_claases = MaskBaseDataset.num_classes #18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.Dataloader(
        dataset,
        batch_size = args.batch_size,
        num_workers= multiprocessing.cpu_count()//2,
        shuffle = False,
        pin_memory=use_cuda,
        drop_last = False
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        save_path = os.path.join(output_dir, f'output.csv')
        info.to_csv(save_path, index=False)
        print(f'Inference Done! Inference result saved at {save_path}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParset()

    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validating (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96,128), help='resize size for image when you trained (default: (96,128))') 
    
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp' ))
    parser.add_argument('--output dir', type=str, deafult=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)

