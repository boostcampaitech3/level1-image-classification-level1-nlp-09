import os, sys
import random
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import preprocess_df, MaskDataset, TestDataset  # preprocess_df_eval,
from dataset_split import CustomMaskSplitByProfileDataset
from dataset_reg import CustomMaskSplitByProfileDatasetReg, preprocess_df_reg
# from utils import AverageMeter, ProgressMeter
import pandas as pd

from utils import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from loss import LabelSmoothingLoss, F1Loss, FocalLoss
import matplotlib.pyplot as plt
import itertools

TRUE_LABELS = ['wear-M-30', 'wear-M-3060', 'wear-M-60',
               'wear-F-30', 'wear-F-3060', 'wear-F-60',
               'incor-M-30', 'incor-M-3060', 'incor-M-60',
               'incor-F-30', 'incor-F-3060', 'incor-F-60',
               'not_wear-M-30', 'not_wear-M-3060', 'not_wear-M-60',
               'not_wear-F-30', 'not_wear-F-3060', 'not_wear-F-60'
               ]


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def plot_confusion_matrix2(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def seed_setting(random_seed):
    '''
    setting random seed for further reproduction
    :param random_seed:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # pytorch, numpy random seed 고정
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # CuDNN 고정
    # torch.backends.cudnn.deterministic = True # 고정하면 학습이 느려진다고 합니다.
    torch.backends.cudnn.benchmark = False
    # GPU 난수 생성
    torch.cuda.manual_seed(random_seed)
    # transforms에서 사용하는 random 라이브러리 고정
    random.seed(random_seed)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def train(model, criterion, optimizer, epoch,
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
    criterion2 = nn.MSELoss() # weight을 주는 기준이 애매함 ㅠㅠ
    cls_loss = 0
    reg_loss = 0
    for img, target, age in tqdm(train_dataloader):
        img, target, age = img.to(device), target.to(device), age.to(device)
        optimizer.zero_grad()
        output, output2 = model(img)

        loss = criterion(output, target)
        loss2 = criterion2(output2.squeeze(), age)
        total_loss = loss+ 0.001*loss2
        total_loss.backward()

        optimizer.step()
        # acc1 = accuracy(output, target)
        _, pred = torch.max(output, 1)
        correct = torch.sum(pred == target)

        cnt += 1
        data_cnt += img.shape[0]
        cls_loss += loss.detach()
        reg_loss += loss2.detach()
        running_loss += total_loss.detach()
        running_acc += correct

        # progress.display_summary()
    print(f'[ Training ]for {epoch}, loss : {running_loss / cnt:.7f} , acc: {running_acc / data_cnt:.7f}')
    return running_loss / cnt, running_acc / data_cnt, cls_loss/cnt, reg_loss/cnt


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
    criterion2 = nn.MSELoss()
    cls_loss = 0
    reg_loss = 0
    model.eval()
    with torch.no_grad():
        for img, target,age in tqdm(val_dataloader):
            img, target, age = img.to(device), target.to(device), age.to(device)
            output, output2 = model(img)
            loss = criterion(output, target)
            loss2 = criterion2(output2.squeeze(), age)
            total_loss = loss + 0.001*loss2

            _, pred = torch.max(output, 1)
            correct = torch.sum(pred == target)
            cnt += 1
            data_cnt += img.shape[0]

            running_loss += total_loss.detach()
            running_acc += correct
            cls_loss += loss.detach()
            reg_loss += loss2.detach()

            val_target.extend(target.cpu().numpy().tolist())
            val_pred.extend(pred.cpu().numpy().tolist())

        f1 = f1_score(val_target, val_pred, average='macro')
        conf = confusion_matrix(val_target, val_pred, labels=list(range(18)))
        fig1 = plot_confusion_matrix(conf, TRUE_LABELS)
        fig2 = plot_confusion_matrix2(conf, TRUE_LABELS)
    early_stopping(running_loss / cnt, model, epoch)

    if early_stopping.early_stop:
        print('!! Requires early stopped !! ')
        early_stop_signal = True

    print(f'[ Val ] for {epoch}, loss: {running_loss / cnt:.7f}, f1/acc: {f1:.7f}, {running_acc / data_cnt:.7f}')
    return running_loss / cnt, running_acc / data_cnt, f1, early_stop_signal, fig1, fig2, cls_loss/cnt, reg_loss/cnt



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
            img = img.to(device)

            output,_ = model(img)
            _, pred = torch.max(output, 1)
            all_predictions.extend(pred.cpu().numpy())
            # correct = torch.sum(pred == target)

            # cnt += img.size(0)

    submission['ans'] = all_predictions
    submission.to_csv(f'{file_name}', index=False)


if __name__ == "__main__":
    ######Configuration: seed, device ######
    random_seed = 42
    seed_setting(random_seed)
    device = 'cuda'

    import argparse

    parser = argparse.ArgumentParser()
    # args.train_batch_size,
    parser.add_argument('--mode', type=str, default='train', help='training/ inference')
    parser.add_argument('--tf_mode', type=str, default='yes', help='training/ inference')

    parser.add_argument('--epochs', type=int, default=20, help='num of epochs')
    parser.add_argument('--model', type=str, default='modified_efficientnet_b3', help='model name')

    parser.add_argument('--img_resize', type=int, default=256, help='model name')
    parser.add_argument('--img_crop', type=int, default=224, help='model name')

    parser.add_argument('--lr', type=float, default=1e-4, help='model name')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--loss', type=str, default='cross_entropy', help='model name')
    parser.add_argument('--alpha', type=float, default=0.1, help='model name')

    args = parser.parse_args()
    print(args)
    ###### Datset ######

    base_dir = '/opt/ml/input/data/train/images'

    # test_df = preprocess_df_eval(base_dir)
    # 14
    # train_transform = transforms.Compose([transforms.Resize(224),
    #                                       # transforms.RandomResizedCrop(224),
    #                                       transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor()])
    # 15 # base 256, 224
    train_transform = transforms.Compose([transforms.Resize(args.img_resize),
                                          transforms.CenterCrop(args.img_crop),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    # add_val_transform 안붙은거는 다 밑에꺼
    # val_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize(args.img_resize),
                                        transforms.CenterCrop(args.img_crop),
                                        transforms.ToTensor()])

    if args.mode == 'inference':
        base_dir = '/opt/ml/input/data/'
        test_image_dir = os.path.join(base_dir, 'eval', 'images')
        total_df, new_train_df_info = preprocess_df(base_dir)

        train_dataset = MaskDataset(base_dir, total_df, transform=train_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)

        submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))
        image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
        test_dataset = TestDataset(image_paths, transform=val_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=4)

    # train_df, val_df = train_test_split(total_df, test_size=0.2,
    #                                 shuffle=True, stratify=total_df['targets'])
    else:
        # total_dataset = CustomMaskSplitByProfileDataset(base_dir, val_ratio=0.2)
        total_dataset = CustomMaskSplitByProfileDatasetReg(base_dir, val_ratio=0.2)
        total_dataset.set_transform(train_transform, val_transform)
        train_dataset, val_dataset = total_dataset.split_dataset()

        # train_dataset = MaskDataset(base_dir, total_df, transform = train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)

        # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                drop_last=True)

    # 만약 best 찾았으면 이걸로 학습시켜야함

    # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    ###### Model ######
    import timm
    import torchvision.models as models

    # model_name ='modified_efficientnet-b0'
    model_name = args.model

    if model_name == 'regnetz_e8':
        model = timm.create_model('regnetz_e8', pretrained=True)
        num_classes = 18
        num_ftrs = 2048
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_classes = 18
        num_ftrs = 512
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.to(device)

    elif model_name == 'resnext50':

        model = models.resnext50_32x4d(pretrained=True)
        num_classes = 18
        num_ftrs = 2048
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.to(device)
    elif model_name == 'regnet_y_800mf':

        model = models.regnet_y_800mf(pretrained=True)
        model.to(device)
    elif model_name == 'modifed_resent18':
        from model import ModifiedResnet18

        model = ModifiedResnet18()
        model.to(device)

    elif model_name == 'efficientnet-b0':
        import timm

        if args.tf_mode == 'yes':
            model = timm.create_model('tf_efficientnet_b0', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b0', pretrained=True)
        num_classes = 18
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.to(device)

    elif model_name == 'efficientnet-b1':
        import timm

        if args.tf_mode == 'yes':
            model = timm.create_model('tf_efficientnet_b1', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b1', pretrained=True)
        num_classes = 18
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.to(device)

    elif model_name == 'efficientnet-b2':
        import timm

        if args.tf_mode == 'yes':
            model = timm.create_model('tf_efficientnet_b2', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b2', pretrained=True)
        num_classes = 18
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.to(device)

    elif model_name == 'efficientnet-b3':
        import timm

        if args.tf_mode == 'yes':
            model = timm.create_model('tf_efficientnet_b3', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b3', pretrained=True)
        num_classes = 18
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.to(device)

    elif model_name == 'efficientnet-b4':
        import timm

        if args.tf_mode == 'yes':
            model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        else:
            model = timm.create_model('efficientnet_b4', pretrained=True)
        num_classes = 18
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.to(device)


    # elif model_name =='modified_efficientnet-b0':
    #     from model import ModifiedEfficientB0
    #     model = ModifiedEfficientB0()
    #     model.to(device)

    elif 'modified_efficientnet' in model_name:
        from model import ModifiedEfficientWithReg

        model = ModifiedEfficientWithReg(args)
        model.to(device)
    # for name,p in model.fc.named_parameters():
    #     if name =='weight':
    #         # nn.init.xavier_normal_(p)
    #         nn.init.xavier_uniform_(p)
    #     elif name =='bias':
    #         nn.init.zeros_(p)
    else:
        raise NotImplementedError

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
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'f1':
        criterion = F1Loss()
    elif args.loss == 'focal':
        # https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
        beta = 0.9999
        train_sample = np.unique(train_dataset.indices, return_counts=True)[1]
        effective_num = 1.0 - np.power(beta, train_sample)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_sample)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=1)

    elif args.loss == 'LDAM':
        from loss import LDAMLoss

        beta = 0.9999
        train_sample = np.unique(train_dataset.indices, return_counts=True)[1]
        effective_num = 1.0 - np.power(beta, train_sample)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_sample)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        criterion = LDAMLoss(cls_num_list=train_sample, max_m=0.5, s=30, weight=per_cls_weights).to(device)
    elif args.loss == 'labelsmoothing':
        criterion = LabelSmoothingLoss(alpha=args.alpha)
    elif args.loss == 'weighted_cross_entropy':

        train_sample = np.unique(train_dataset.indices, return_counts=True)[1]
        normedW = [1 - (x / sum(train_sample)) for x in train_sample]
        normedW = torch.FloatTensor(normedW).to(device)
        criterion = nn.CrossEntropyLoss(weight=normedW)

    # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,nesterov=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                                            lr_lambda=lambda epoch: 0.995 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    # opt = optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = args.epochs

    min_loss = 1e9
    min_acc = 1e9
    # if not uniform -> normal
    # writer = SummaryWriter("tb_split/init_xavier_uniform_CE_val_transform")
    writer_name = f"{args.loss}_{args.lr}{args.alpha}_labelmodified_normalsampling_BS{args.train_batch_size}_Adam"

    if args.mode == 'inferece':
        writer_name = 'FULL_' + writer_name

    if args.tf_mode == 'yes':
        new_model_name = 'tf_' + args.model
    else:
        new_model_name = args.model

    writer = SummaryWriter(f"tb_reg_report/conf_tests/{args.mode}/{new_model_name}/{writer_name}")
    early_stopping = EarlyStopping(patience=8,
                                   base_dir='./results_split/',
                                   file_name=f'{args.mode}_{new_model_name}_{writer_name}')
    for epoch in tqdm(range(num_epochs)):

        train_loss, train_acc, cls_loss, reg_loss = train(model, criterion,
                                      opt, epoch,
                                      train_loader,
                                      device)
        if args.mode != 'inference':
            val_loss, val_acc, val_f1, early_stop_signal, fig1, fig2, val_cls_loss, val_reg_loss = eval(model, criterion,
                                                                            epoch, early_stopping,
                                                                            val_loader)

            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/acc', train_acc, epoch)
            writer.add_scalar('Train/cls_loss', cls_loss, epoch)
            writer.add_scalar('Train/reg_loss', reg_loss, epoch)
            # writer.add_scalar('Train/f1', train_acc, epoch)

            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/acc', val_acc, epoch)
            writer.add_scalar('Val/f1', val_f1, epoch)
            writer.add_figure('Val/conf_based_on_sample_nums', fig1, epoch)
            writer.add_figure('Val/conf_based_on_ratio', fig2, epoch)
            writer.add_scalar('Train/val_cls_loss', val_cls_loss, epoch)
            writer.add_scalar('Train/val_reg_loss', val_reg_loss, epoch)

        else:
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/acc', train_acc, epoch)

            torch.save({'model': model.state_dict(),
                        'loss': train_loss,
                        'optimizer': opt.state_dict()},
                       f'./ckpt_split/{args.mode}/{new_model_name}/{epoch}_{writer_name}.pt')
            if epoch > 3:
                test(model, test_loader,
                     file_name=f'./results_split/{args.mode}/{new_model_name}/{epoch}_{writer_name}.csv')
        print(f'Training finished learning rate at {opt.param_groups[0]["lr"]}')

        early_stop_signal = False
        if early_stop_signal:
            print(f'Therefore finishing training at {epoch}')
            torch.save({'model': model.state_dict(),
                        'loss': train_loss,
                        'optimizer': opt.state_dict()},
                       f'./ckpt_split/{args.mode}/{new_model_name}/{epoch}_{writer_name}.pt')
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
