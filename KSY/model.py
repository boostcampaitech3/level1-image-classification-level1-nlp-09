# test junseok

import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ModifiedResnet18(nn.Module):
    def __init__(self,num_classes=18):
        super(ModifiedResnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = 512
        self.resnet.fc = nn.Linear(num_ftrs, 256)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

class ModifiedEfficient(nn.Module):
    def __init__(self,args, num_classes=18):
        super(ModifiedEfficient, self).__init__()
        """
        fc2 layer, dropout, relu 추가
        """
        model_size = args.model[-1]
        self.efficient = timm.create_model(f'efficientnet_b{model_size}',pretrained=True)
        num_ftrs = self.efficient.classifier.in_features
        self.efficient.classifier = nn.Linear(num_ftrs, 512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.efficient(input)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

class ModifiedEfficientB0(nn.Module):
    def __init__(self,num_classes=18):
        super(ModifiedEfficientB0, self).__init__()
        self.efficient = timm.create_model('efficientnet_b0',pretrained=True)
        num_ftrs = 1280
        self.efficient.classifier = nn.Linear(num_ftrs, 512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.efficient(input)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

if __name__=='__main__':
    mod = ModifiedEfficient(1)