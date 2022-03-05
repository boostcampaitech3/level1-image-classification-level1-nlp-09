import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from facenet_pytorch import MTCNN, InceptionResnetV1
from res_mlp_pytorch import ResMLP
from efficientnet_pytorch import EfficientNet
import timm
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.net.fc.weight)
        stdv = 1. / math.sqrt(self.net.fc.weight.size(1))
        self.net.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.feature_extractor = EfficientNet.from_pretrained("efficientnet-b5")
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class NfnetModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.feature_extractor = timm.create_model("eca_nfnet_l2",
                num_classes=num_classes,
                pretrained = True)
    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class ModifiedEfficientB0(nn.Module) :
    def __init__(self, num_classes=18):
        super(ModifiedEfficientB0,self).__init__()

        self.efficient = timm.create_model('efficientnet_b0',pretrained=True)
        num_ftrs=1280
        self.efficient.classifier = nn.Linear(num_ftrs,512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512,num_classes)
        self.relu = nn.ReLU()

    def forward(self,input):
        x = self.efficient(input)
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        from torchvision.models import vgg19_bn
        self.model = vgg19_bn(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# Custom Model Template
class MyModel_gan(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        G = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU,
            nn.Linear(256,784),
            nn.Tanh()
        )
        D = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x