import argparse
import os, sys
import random
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import timm
import torchvision.models as models
from model import ModifiedResnet18
from model import ModifiedEfficient


# from dataset import preprocess_df, MaskDataset, TestDataset, CustomMaskSplitByProfileDataset
from dataset_fold import preprocess_df, MaskDataset, TestDataset, CustomMaskSplitByProfileDatasetFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from loss import LabelSmoothingLoss, F1Loss, FocalLoss
from utils import EarlyStopping

TRUE_LABELS = ['wear-M-30', 'wear-M-3060', 'wear-M-60',
               'wear-F-30', 'wear-F-3060', 'wear-F-60',
               'incor-M-30', 'incor-M-3060', 'incor-M-60',
               'incor-F-30', 'incor-F-3060', 'incor-F-60',
               'not_wear-M-30', 'not_wear-M-3060', 'not_wear-M-60',
               'not_wear-F-30', 'not_wear-F-3060', 'not_wear-F-60'
               ]


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    confunsion matrix 리스트로 받아서 sample 갯수에 따라 강도를 나타내는 그림
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
    confunsion matrix 리스트로 받아서 해당 클래스의 갯수/ 전체 갯수로 normalization해서 ratio에 따라 density 표시
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


def train(model, criterion, optimizer, epoch,
          train_dataloader,
          device='cuda'
          ):
    """
    훈련 함수
    """
    cnt = 0
    data_cnt = 0
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for img, target in tqdm(train_dataloader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output, 1)
        correct = torch.sum(pred == target)

        cnt += 1
        data_cnt += img.shape[0]

        running_loss += loss.detach()
        running_acc += correct

    print(f'[ Training ]for {epoch}, loss : {running_loss / cnt:.7f} , acc: {running_acc / data_cnt:.7f}')
    return running_loss / cnt, running_acc / data_cnt


def eval(model, criterion, epoch, early_stopping,
         val_dataloader,
         device='cuda',
         ):
    """
    valiation 데이터로더가 있을 경우 validation 하는 함수)
    """
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
            _, pred = torch.max(output, 1)
            correct = torch.sum(pred == target)
            cnt += 1
            data_cnt += img.shape[0]

            running_loss += loss.detach()
            running_acc += correct

            val_target.extend(target.cpu().numpy().tolist())
            val_pred.extend(pred.cpu().numpy().tolist())

        f1 = f1_score(val_target, val_pred, average='macro')
        conf = confusion_matrix(val_target, val_pred, labels=list(range(18)))
        fig1 = plot_confusion_matrix(conf, TRUE_LABELS)
        fig2 = plot_confusion_matrix2(conf, TRUE_LABELS)

    # validation 로스 기준으로 early stopping
    early_stopping(running_loss / cnt, model, epoch)

    if early_stopping.early_stop:
        print('!! Requires early stopped !! ')
        early_stop_signal = True

    print(f'[ Val ] for {epoch}, loss: {running_loss / cnt:.7f}, f1/acc: {f1:.7f}, {running_acc / data_cnt:.7f}')
    return running_loss / cnt, running_acc / data_cnt, f1, early_stop_signal, fig1, fig2


def test(model,
         test_dataloader,args,
         file_name,
         device='cuda',
         ):
    """
    submission용 inference 수행 함수
    """

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for img in tqdm(test_dataloader):
            img = img.to(device)

            output = model(img)
            if args.TTA=='use':
                # 원본 이미지를 예측
                pred = model(img) / 2
                pred += model(torch.flip(img, dims=(-1,))) / 2 # horizontal?
            else:
                _, pred = torch.max(output, 1)

            all_predictions.extend(pred.cpu().numpy())
        fold_pred = np.array(all_predictions)
    return fold_pred


if __name__ == "__main__":

    ######Configuration: seed, device ######
    random_seed = 42
    seed_setting(random_seed)
    device = 'cuda'

    parser = argparse.ArgumentParser()
    # args.train_batch_size,
    parser.add_argument('--mode', type=str, default='train', help='training/ inference')
    parser.add_argument('--tf_mode', type=str, default='no', help='efficientnet test- use tf version')

    parser.add_argument('--epochs', type=int, default=15, help='num of epochs')
    parser.add_argument('--model', type=str, default='modified_efficientnet-b3 ', help='model name')

    parser.add_argument('--img_resize', type=int, default=332, help='image resizing')
    parser.add_argument('--img_crop', type=int, default=300, help='image cropping')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='batch size, if using b4-> using batch size 32')

    parser.add_argument('--loss', type=str, default='cross_entropy', help='type of loss')
    parser.add_argument('--alpha', type=float, default=0.1, help='smoothing for label smoothing')

    parser.add_argument('--TTA', type=str, default='use', help='choose wheter to use TTA')
    parser.add_argument('--k_split', type=int, default=5, help='choose wheter to use TTA')
    parser.add_argument('--patience', type=int, default=8, help='choose wheter to use TTA')

    args = parser.parse_args()
    print(args)
    print(f'지금 돌리는 모드는 {args.mode} 입니다! 모델은 {args.model}이고 인풋 사이즈는 {args.img_resize}로 줄이고, {args.img_crop}만큼 짜릅니당')

    ###### Datset ######

    base_dir = '/opt/ml/input/data/train/images'
    """
    Randomness가 들어가지 않은 transform은 val_transform에 넣는 것이 실험적으로 나음
    """
    train_transform = transforms.Compose([transforms.Resize(args.img_resize),
                                          transforms.CenterCrop(args.img_crop),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.Resize(args.img_resize),
                                        transforms.CenterCrop(args.img_crop),
                                        transforms.ToTensor()])

    """
    @준석: dataset.py의 MaskDataset 클래스의 __getitem__과 Subset의 __getitem__에서 transformation 수행됨

    """
    if args.mode == 'inference':
        """
        Not used
        
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
        """
        pass
    else:
        """
        Validation을 위해 val_ratio 만큼 전체 데이터를 train:val=(1-val_ratio):val_ratio 로 잘라서 학습할 수 있도록
        trainloader, val_loader 생성.
        전체 이미지 기준으로 split할 경우 cheating 가능성 존재 -> 폴더기준으로 split하는 Dataset을 기본으로 삼음.
        단, 주어진 베이스라인 코드 경우 train, val transform이 동일하게 적용되기 때문에 dataset.py에서 Subset 클래스로 각각 transform 설정될 수 있게 수정함
        """
        """
        Fold는 train mode로 다 수행
        """
        total_dataset_fold = CustomMaskSplitByProfileDatasetFold(base_dir, val_ratio=0.2,
                                                            k_split= args.k_split)
        total_dataset_fold.set_transform(train_transform, val_transform)
        train_dataset_fold, val_dataset_fold = total_dataset_fold.split_dataset()

        # # train_dataset = MaskDataset(base_dir, total_df, transform = train_transform)
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
        #                           drop_last=True)
        #
        # # val_dataset = MaskDataset(base_dir, val_df, transform=val_transform)
        # val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
        #                         drop_last=True)

        base_dir = '/opt/ml/input/data/'
        test_image_dir = os.path.join(base_dir, 'eval', 'images')
        submission = pd.read_csv(os.path.join(base_dir, 'eval', 'info.csv'))
        image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]
        test_dataset = TestDataset(image_paths, transform=val_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=4)

    ###### loss, opt ######
    if args.loss == 'cross_entropy':
        """
        기본: cross_entropy
        """
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

    """
    K-fold done
    """
    oof_pred = None
    for fold_idx, (train_dataset, val_dataset) in enumerate(zip(train_dataset_fold, val_dataset_fold)):
        ###### Model ######
        # model_name ='modified_efficientnet-b3'
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
            breakpoint()
            model = models.regnet_y_800mf(pretrained=True)
            model.to(device)
        elif model_name == 'modifed_resent18':
            model = ModifiedResnet18()
            model.to(device)

        elif model_name == 'efficientnet-b0':
            if args.tf_mode == 'yes':
                model = timm.create_model('tf_efficientnet_b0', pretrained=True)
            else:
                model = timm.create_model('efficientnet_b0', pretrained=True)
            num_classes = 18
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.to(device)

        elif model_name == 'efficientnet-b1':
            if args.tf_mode == 'yes':
                model = timm.create_model('tf_efficientnet_b1', pretrained=True)
            else:
                model = timm.create_model('efficientnet_b1', pretrained=True)
            num_classes = 18
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.to(device)

        elif model_name == 'efficientnet-b2':

            if args.tf_mode == 'yes':
                model = timm.create_model('tf_efficientnet_b2', pretrained=True)
            else:
                model = timm.create_model('efficientnet_b2', pretrained=True)
            num_classes = 18
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.to(device)

        elif model_name == 'efficientnet-b3':

            if args.tf_mode == 'yes':
                model = timm.create_model('tf_efficientnet_b3', pretrained=True)
            else:
                model = timm.create_model('efficientnet_b3', pretrained=True)
            num_classes = 18
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.to(device)

        elif model_name == 'efficientnet-b4':
            if args.tf_mode == 'yes':
                model = timm.create_model('tf_efficientnet_b4', pretrained=True)
            else:
                model = timm.create_model('efficientnet_b4', pretrained=True)
            num_classes = 18
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model.to(device)

        elif 'modified_efficientnet' in model_name:
            """
            버전에 따라서 ModifiedEfficient 클래스에서 버전에 맞게 return
            """
            model = ModifiedEfficient(args)
            model.to(device)

        else:
            print('설정한 모델이 없는디용?')
            raise NotImplementedError

        # 만약 optimizer 바꿀거면 바꿔주세요
        opt_name = 'Adam'  # opt_name으로 tensorboard writer 이름 들어감
        opt = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                                                lr_lambda=lambda epoch: 0.995 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)

        num_epochs = args.epochs
        min_loss = 1e9
        min_acc = 1e9
        writer_name = f"{args.loss}_{args.lr}{args.alpha}_normalsampling_BS{args.train_batch_size}_{opt_name}"

        if args.mode == 'inferece':
            writer_name = 'FULL_' + writer_name

        if args.tf_mode == 'yes':
            new_model_name = 'tf_' + args.model
        else:
            new_model_name = args.model

        # writer = SummaryWriter(f"tb_report/conf_tests/{args.mode}/{new_model_name}/{writer_name}")
        writer = SummaryWriter(f"tb_report/conf_tests/{args.mode}/{new_model_name}/{fold_idx}_{writer_name}")
        ###### submission file 및 ckpt 저장 Directory 생성 ######
        # os.makedirs(f'./ckpt_split/{args.mode}/{new_model_name}/')
        # os.makedirs(f'./results_split/{args.mode}/{new_model_name}/')
        mkdirs(f'./ckpt_split/{args.mode}/{new_model_name}/')
        mkdirs(f'./results_split/{args.mode}/{new_model_name}/')
        # early stopping, patience=5 의 의미: 최저 val_loss 기준으로 5epoch까지만 봐줌
        early_stopping = EarlyStopping(patience=args.patience,
                                       base_dir='./results_split/',
                                       file_name=f'{args.mode}_{new_model_name}_{writer_name}')

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)

        val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                drop_last=True)

        for epoch in tqdm(range(num_epochs)):
            """
            K-fold : 각 K에 맞는 train_dataset에 따라 train_loader, val_loader만들어짐
            args.mode가 inference :학습 후 test 함수로 eval 데이터셋에 대해 inference 실행
            args.mode 가 train : 학습 후 eval 함수로 전체 데이터셋에서 쪼갠 validation loader 데이터에 대해 validation 수행
            """
            train_loss, train_acc = train(model, criterion,
                                          opt, epoch,
                                          train_loader,
                                          device)
            if args.mode != 'inference':
                val_loss, val_acc, val_f1, early_stop_signal, fig1, fig2 = eval(model, criterion,
                                                                                epoch, early_stopping,
                                                                                val_loader)

                writer.add_scalar('Train/loss', train_loss, epoch)
                writer.add_scalar('Train/acc', train_acc, epoch)
                # writer.add_scalar('Train/f1', train_acc, epoch)

                writer.add_scalar('Val/loss', val_loss, epoch)
                writer.add_scalar('Val/acc', val_acc, epoch)
                writer.add_scalar('Val/f1', val_f1, epoch)
                writer.add_figure('Val/conf_based_on_sample_nums', fig1, epoch)
                writer.add_figure('Val/conf_based_on_ratio', fig2, epoch)
            else:
                """
                Not used here
                writer.add_scalar('Train/loss', train_loss, epoch)
                writer.add_scalar('Train/acc', train_acc, epoch)
    
                torch.save({'model': model.state_dict(),
                            'loss': train_loss,
                            'optimizer': opt.state_dict()},
                           f'./ckpt_split/{args.mode}/{new_model_name}/{epoch}_{writer_name}.pt')
                if epoch > 3:
                    test(model, test_loader,
                         file_name=f'./results_split/{args.mode}/{new_model_name}/{epoch}_{writer_name}.csv')
                """
                pass
            print(f'{fold_idx} model at {epoch}: Training finished learning rate at {opt.param_groups[0]["lr"]}')
            # scheduler step
            scheduler.step()
            # early stop 돼면 마지막꺼를 저장
            if epoch>4:
                if early_stop_signal:
                    print(f'{fold_idx} requires early stopped! finishing training at {epoch}')
                    torch.save({'model': model.state_dict(),
                                'loss': train_loss,
                                'optimizer': opt.state_dict()},
                               f'./ckpt_split/{args.mode}/{new_model_name}/{fold_idx}model_{epoch}_{writer_name}.pt')
            if early_stop_signal:
                print(f'{fold_idx} requires early stopped! finishing training at {epoch}')
                torch.save({'model': model.state_dict(),
                            'loss': train_loss,
                            'optimizer': opt.state_dict()},
                           f'./ckpt_split/{args.mode}/{new_model_name}/{fold_idx}model_{epoch}_{writer_name}.pt')
                fold_pred = test(model, test_loader,args,
                     file_name=f'./results_split/{args.mode}/{new_model_name}/{fold_idx}model_{epoch}_{writer_name}.csv')
                break
        if oof_pred is None:
            oof_pred = fold_pred / args.k_split
        else:
            oof_pred += fold_pred / args.k_split

    submission['ans'] = np.argmax(oof_pred, axis=1)
    save_dir ='/opt/ml/level1-image-classification-level1-nlp-09/KSY/'
    submission.to_csv(os.path.join(save_dir, f'{args.k_split}_{args.model}_submission.csv'), index=False)

