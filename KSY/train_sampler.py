import argparse

import os, sys
import random
import time
import torch
from torch import optim,nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import preprocess_df,preprocess_df_eval, MaskDataset, TestDataset
from dataset_split import CustomMaskSplitByProfileDataset
# from utils import AverageMeter, ProgressMeter
import pandas as pd

from utils import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from loss import LabelSmoothingLoss, F1Loss, FocalLoss

from utils import ImbalancedDatasetSampler


def seed_setting(random_seed):
    '''
    setting random seed for further reproduction
    :param random_seed:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #pytorch, numpy random seed 고정
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # CuDNN 고정
    # torch.backends.cudnn.deterministic = True # 고정하면 학습이 느려진다고 합니다.
    torch.backends.cudnn.benchmark = False
    # GPU 난수 생성
    torch.cuda.manual_seed(random_seed)
    #transforms에서 사용하는 random 라이브러리 고정
    random.seed(random_seed)


def train(model, criterion, optimizer,epoch,
        train_dataloader,
          device='cuda'
          ):


    cnt = 0
    data_cnt = 0
    running_loss = 0.0
    running_acc = 0.0

    train_loss_all = []
    train_acc_all = []

    model.train()

    for img, target in tqdm(train_dataloader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # acc1 = accuracy(output, target)
        _, pred = torch.max(output, 1)
        correct = torch.sum(pred == target)

        cnt += 1
        data_cnt += img.shape[0]

        running_loss += loss.detach()
        running_acc += correct

        # progress.display_summary()
    print(f'[ Training ]for {epoch}, loss : {running_loss/cnt:.7f} , acc: {running_acc/data_cnt:.7f}')
    return running_loss/cnt, running_acc/data_cnt

def eval(model, criterion, epoch, early_stopping,
        val_dataloader,
          device='cuda',
          ):
    # base_dir = '/opt/ml/input/data'
    # submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))

    cnt = 0
    data_cnt = 0
    running_loss = 0.0
    running_acc = 0.0

    val_target = []
    val_pred = []
    early_stop_signal = False
    model.eval()
    with torch.no_grad():
        for img, target in tqdm(val_dataloader):
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss = criterion(output, target)
            _, pred = torch.max(output,1)
            correct = torch.sum(pred == target)
            cnt += 1
            data_cnt += img.shape[0]

            running_loss += loss.detach()
            running_acc += correct

            val_target.extend(target.cpu().numpy().tolist())
            val_pred.extend(pred.cpu().numpy().tolist())

        f1 = f1_score(val_target, val_pred, average='macro')
    early_stopping(running_loss/cnt, model, epoch)

    if early_stopping.early_stop:
        print('!! Requires early stopped !! ')
        early_stop_signal= True

    print(f'[ Val ] for {epoch}, loss: {running_loss/cnt:.7f}, f1/acc: {f1:.7f}, {running_acc/data_cnt:.7f}')
    return running_loss/cnt, running_acc/data_cnt, f1, early_stop_signal

def test(model,
          test_dataloader,
            file_name,
          device='cuda',
          ):
    base_dir = '/opt/ml/input/data'
    submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))

    model.eval()
    all_predictions = []
    with torch.no_grad():

        for img in tqdm(test_dataloader):
            img= img.to(device)

            output = model(img)
            _, pred = torch.max(output,1)
            all_predictions.extend(pred.cpu().numpy())
            # correct = torch.sum(pred == target)

            # cnt += img.size(0)

    submission['ans'] = all_predictions
    submission.to_csv(f'{file_name}',index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=train, help='training/ inference')
    parser.add_argument('--epochs', type=int, default=20, help='training/ inference')
    args = parser.parse_args()

    ######Configuration: seed, device ######
    random_seed = 42
    seed_setting(random_seed)
    device = 'cuda'


    ###### Datset ######

    base_dir = '/opt/ml/input/data/train/images'

    # test_df = preprocess_df_eval(base_dir)

    input_size = 256
    # 14
    # train_transform = transforms.Compose([transforms.Resize(224),
    #                                       # transforms.RandomResizedCrop(224),
    #                                       transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor()])
    # 15
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    # add_val_transform 안붙은거는 다 밑에꺼
    # val_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                        transforms.ToTensor()])

    if args.mode =='inference':
        base_dir = '/opt/ml/input/data/'
        test_image_dir = os.path.join(base_dir, 'eval', 'images')
        total_df,new_train_df_info = preprocess_df(base_dir)

        train_dataset = MaskDataset(base_dir, total_df, transform=train_transform)

        train_loader = DataLoader(train_dataset,
                                  sampler=ImbalancedDatasetSampler(train_dataset, mode=args.mode),
                                  batch_size=64,
                                  num_workers=4,
                                  drop_last=True)

        """
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
        """
        submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))
        image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
        test_dataset = TestDataset(image_paths, transform=val_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=4)

    # train_df, val_df = train_test_split(total_df, test_size=0.2,
    #                                 shuffle=True, stratify=total_df['targets'])
    else:


        total_dataset = CustomMaskSplitByProfileDataset(base_dir, val_ratio=0.2)
        total_dataset.set_transform(train_transform, val_transform)
        train_dataset, val_dataset = total_dataset.split_dataset()

        # train_dataset = MaskDataset(base_dir, total_df, transform = train_transform)

        train_loader = DataLoader(train_dataset,
                                  sampler=ImbalancedDatasetSampler(train_dataset),
                                  batch_size=64,

                                  num_workers=4,
                                  drop_last=True)
        """
        train_loader = DataLoader(train_dataset, 
                                  batch_size=64, 
                                  shuffle=True, 
                                  num_workers=4,
                                  drop_last=True)
        """

        # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)


    # 만약 best 찾았으면 이걸로 학습시켜야함


    # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    ###### Model ######
    # model = models.resnet18(pretrained=True)
    # num_classes = 18
    # num_ftrs = 512
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.to(device)
    from model import ModifiedResnet18, ModifiedEfficientB0

    # model = ModifiedResnet18()
    model = ModifiedEfficientB0()
    model.to(device)
    # for name,p in model.fc.named_parameters():
    #     if name =='weight':
    #         # nn.init.xavier_normal_(p)
    #         nn.init.xavier_uniform_(p)
    #     elif name =='bias':
    #         nn.init.zeros_(p)



    # To finetune : https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # params_to_update = model_ft.parameters()
    # print("Params to learn:")
    # if feature_extract:
    #     params_to_update = []
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             params_to_update.append(param)
    #             print("\t",name)
    # else:
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)

    ###### loss, opt ######
    from loss import LabelSmoothingLoss
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(smoothing=0.1)
    # criterion = F1Loss()
    # criterion = FocalLoss()

    # train_sample = np.unique(train_dataset.indices, return_counts=True)[1]
    # normedW = [1 - (x / sum(train_sample)) for x in train_sample]
    # normedW = torch.FloatTensor(normedW).to(device)
    # criterion =  nn.CrossEntropyLoss(weight=normedW)

    # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,nesterov=True)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                                            lr_lambda=lambda epoch: 0.995 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    # opt = optim.Adam(model.parameters(), lr=3e-4)

    min_loss = 1e9
    min_acc = 1e9
    # if not uniform -> normal
    # writer = SummaryWriter("tb_split/init_xavier_uniform_CE_val_transform")
    writer_name = "weighted_train_sampler_modifed_efficientb0"
    writer = SummaryWriter(f"tb_split/semi_final/{writer_name}")
    early_stopping = EarlyStopping(patience = 8,
                                   base_dir = './results_split/',
                                   file_name=writer_name)
    for epoch in tqdm(range(args.epochs)):

        train_loss, train_acc = train(model, criterion,
                                      opt,epoch,
                                      train_loader,
                                      device)
        if args.mode != 'inference':
            val_loss, val_acc, val_f1, early_stop_signal = eval(model, criterion,
                                    epoch,early_stopping,
                                    val_loader)

            writer.add_scalar('Train/loss',train_loss, epoch)
            writer.add_scalar('Train/acc', train_acc, epoch)
            # writer.add_scalar('Train/f1', train_acc, epoch)

            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/acc', val_acc, epoch)
            writer.add_scalar('Val/f1', val_f1, epoch)
        else:
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/acc', train_acc, epoch)

            torch.save({'model': model.state_dict(),
                        'loss': train_loss,
                        'optimizer': opt.state_dict()}, f'./ckpt_split/{writer_name}_{epoch}.pt')
            if epoch >3:
                test(model, test_loader, file_name=f'./results_split/{writer_name}_{epoch}.csv')
        print(f'Training finished learning rate at {opt.param_groups[0]["lr"]}')

        early_stop_signal = False
        if early_stop_signal:
            print(f'Therefore finishing training at {epoch}')
            torch.save({'model':model.state_dict(),
                        'loss':train_loss,
                        'optimizer':opt.state_dict()}, f'./ckpt_split/{writer_name}_{epoch}.pt')
            # test(model, test_loader,file_name=f'./results_split/aug_adam{epoch}_2.csv')

        scheduler.step()
        # if (epoch+1) > 8 :
        #     if min_loss > train_loss:
        #         min_loss = train_loss
        #         #first sub : momentum : 0.001
        #         torch.save({'model':model.state_dict(),
        #                     'loss':train_loss,
        #                     'optimizer':opt.state_dict()}, f'./ckpt_split/aug_res18_momentumSGD_0.01{epoch}_2.pt')
        #         test(model, test_loader,file_name=f'./results_split/aug_res18_momentumSGD_0.01{epoch}_2.csv')
