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
from utils import AverageMeter, ProgressMeter
import pandas as pd

from sklearn.model_selection import train_test_split

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
        train_dataloader, val_dataloader=None,
          device='cuda'
          ):
    # ref : https://github.com/pytorch/examples/blob/master/imagenet/main.py
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_dataloader),
    #     [batch_time, data_time, losses, top1],
    #     prefix="Epoch: [{}]".format(epoch))

    # end = time.time()

    # data_time.update(time.time() - end)


    running_loss = 0.0
    cnt = 0
    running_acc = 0.0
    model.train()
    # class_cnts = {k:0 for k in range(18)}
    # {0: 1080, 1: 1115, 2: 1082, 3: 1010, 4: 1040, 5: 1067, 6: 1001, 7: 1060, 8: 1041, 9: 1039, 10: 1032, 11: 1087, 12: 1027, 13: 1071, 14: 1056, 15: 1040, 16: 1036, 17: 996}
    for img, target in tqdm(train_dataloader):
        img, target = img.to(device), target.to(device)

        taregt_cls,target_stats = torch.unique(target, return_counts=True)
        # for (t_cls, t_st) in zip(taregt_cls,target_stats):
        #     class_cnts[t_cls.item()] += t_st.item()

        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # acc1 = accuracy(output, target)
        _, pred = torch.max(output, 1)
        correct = torch.sum(pred == target)

        cnt += img.size(0)
        running_loss += loss.detach()

        running_acc += correct
        # losses.update(loss.item(), img.size(0))
        # top1.update(acc1[0], img.size(0))
        # batch_time.update(time.time() - end)
        # end = time.time()

        # progress.display_summary()

    print(f'[ Training ]for {epoch}, loss : {running_loss/cnt} , acc: {running_acc/cnt}')
    return running_loss/cnt, running_acc/cnt
    # eval(model, epoch, val_dataloader)
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
    submission.to_csv(f'./{file_name}',index=False)

def eval(model, epoch,
          val_dataloader,
          device='cuda',
          ):
    base_dir = '/opt/ml/input/data'
    submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))
    cnt = 0
    running_acc = 0.0

    model.eval()
    all_predictions = []
    with torch.no_grad():
        print('val')
        for img, target in tqdm(val_dataloader):
            img, target = img.to(device), target.to(device)

            output = model(img)
            _, pred = torch.max(output,1)
            all_predictions.extend(pred.cpu().numpy())
            # correct = torch.sum(pred == target)
            submission['ans'] = all_predictions
            # cnt += img.size(0)

    submission.to_csv('./test.csv',index=False)

            # running_acc += correct

    # print(f'[ Val ] for {epoch}, acc: {running_acc / cnt}')

if __name__=="__main__":
    ######Configuration: seed, device ######
    random_seed = 42
    seed_setting(random_seed)
    device = 'cuda'


    ###### Datset ######

    base_dir = '/opt/ml/input/data'
    total_df, total_df_info = preprocess_df(base_dir)
    # test_df = preprocess_df_eval(base_dir)

    ###### Sampler ######
    """
    from torch.utils.data import WeightedRandomSampler

    num_class_samples = np.array([total_df_info[k] for k in sorted(total_df_info)])
    weights = 1. / num_class_samples
    samples_weight = np.array([weights[total_df.iloc[idx]['targets']] for idx in range(len(total_df))])

    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    """

    
    test_image_dir = os.path.join(base_dir,'eval','images')

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
    val_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    # train_df, val_df = train_test_split(total_df, test_size=0.2,
    #                                 shuffle=True, stratify=total_df['targets'])

    train_dataset = MaskDataset(base_dir, total_df, transform = train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, #shuffle=True,
                              sampler=sampler,
                              num_workers=4,drop_last=True)

    submission = pd.read_csv(os.path.join(base_dir,'eval', 'info.csv'))
    image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(image_paths, transform = val_transform)
    test_loader = DataLoader(test_dataset,  shuffle=False, batch_size=32, num_workers=4)

    # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    ###### Model ######

    model = models.resnet18(pretrained=True)
    num_classes = 18
    num_ftrs = 512
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

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
    criterion = nn.CrossEntropyLoss()
    # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,nesterov=True)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                                            lr_lambda=lambda epoch: 0.995 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    # opt = optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = 5

    min_loss = 1e9
    min_acc = 1e9
    test_mode = False
    if test_mode:
        writer = SummaryWriter("tests")
    else:
        writer = SummaryWriter("sampler_tests/")
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train(model, criterion, opt,epoch,
              train_loader, test_loader,
              device)
        writer.add_scalar('Loss/train',train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        scheduler.step()
        if (epoch+1) > 4 :
            if min_loss > train_loss:
                min_loss = train_loss
                #first sub : momentum : 0.001
                torch.save({'model':model.state_dict(),
                            'loss':train_loss,
                            'optimizer':opt.state_dict()}, f'./ckpt/Resnet18_Adam1e-4_weighted_sampler_Half_{epoch}.pt')
                test(model, test_loader,file_name=f'./results/Resnet18_Adam1e-4_weighted_sampler_Half_{epoch}.csv')
