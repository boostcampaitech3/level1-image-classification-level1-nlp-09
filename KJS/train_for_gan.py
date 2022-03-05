import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import sys
from importlib import import_module
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import torch.nn as nn

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from torchsampler import ImbalancedDatasetSampler


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    # train_set, val_set = dataset.split_dataset()
    train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=args.valid_batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )
    ######################################################
    G = nn.Sequential(
        nn.Linear(128,96),
        nn.ReLU(),
        nn.Linear(96,96),
        nn.ReLU(),
        nn.Linear(96,50176),
        nn.Tanh())

    # 판별자 (Discriminator)
    # Leaky ReLU : 약간의 음의 기울기도 다음 layer로 전달
    # discriminator에서 계산한 기울기가 0이 아닌 약한 음수로 전환되며 생성자 측에 더 강하게 전달

    D = nn.Sequential(
        nn.Linear(50176, 96),
        nn.LeakyReLU(0.2),
        nn.Linear(96,96),
        nn.LeakyReLU(0.2),
        nn.Linear(96, 1),
        nn.Sigmoid())

    # 모델의 가중치를 지정한 장치로 보내기
    # CUDA(GPU) / CPU
    DEVICE = "cuda"
    D = D.to(DEVICE)
    G = G.to(DEVICE)

    # 이진 크로스 엔트로피 (Binary cross entropy) 오차 함수와
    # 생성자와 판별자를 최적화할 Adam 모듈
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

    EPOCHS = 10000000
    BATCH_SIZE = 96

    # print(train_set)
    total_step = len(train_loader)

    # print(train_loader)
    for epoch in tqdm(range(EPOCHS)):
        for i, (images, _) in tqdm(enumerate(train_loader)):
            images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
            # images = images.view(BATCH_SIZE, -1).to(DEVICE)

            # print('i :',i)
            # print('_ :', _)
            # print('images :', images)

            # 이미지 정규화
            # images = (images - 127.5) / 127.5
            # 이미지의 사이즈를 알아야지??
            # print(type(images))
            # print(images.ndim())
            # print(images.shape)
            # print(images.size)

            # '진짜'와 '가짜' 레이블 생성
            real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)  # [1,1,1...]
            fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)  # [0.0,0...]

            # 판별자가 진짜 이미지를 진짜로 인식하는 오차를 예산
            outputs = D(images)  # 진짜 이미지를 discriminator의 입력으로 제공
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # 무작위 텐서로 가짜 이미지 생성
            z = torch.randn(BATCH_SIZE, 128).to(DEVICE)
            fake_images = G(z)  # G의 입력으로 랜덤 텐서 제공, G가 fake image 생성

            # 판별자가 가짜 이미지를 가짜로 인식하는 오차를 계산
            outputs = D(fake_images)  # 가짜 이미지를 discriminator의 입력으로 제공
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 Discriminator의 오차 계산
            d_loss = d_loss_real + d_loss_fake

            # ------ Discriminator 학습 ------#
            # 역전파 알고리즘으로 Discriminator의 학습을 진행
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()  # Discriminator 학습

            # 생성자가 판별자를 속였는지에 대한 오차(Generator의 loss)를 계산
            fake_images = G(z)
            outputs = D(fake_images)  # 한번 학습한 D가 fake image를
            g_loss = criterion(outputs, real_labels)

            # ------ Generator 학습 ------#

            # 역전파 알고리즘으로 생성자 모델의 학습을 진행
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # 학습 진행 알아보기
        print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
              .format(epoch, EPOCHS, d_loss.item(), g_loss.item(),
                      real_score.mean().item(), fake_score.mean().item()))

        # 생성 결과물 확인
        z = torch.randn(BATCH_SIZE, 128).to(DEVICE)
        fake_images = G(z)
        for i in range(1):
            fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (224, 224))
            # plt.imshow(fake_images_img)
            plt.imshow(fake_images_img, cmap='gray')
            plt.savefig(f'/opt/ml/v2/output/gan99/epoch_{epoch}_test_gan{i}.png')
            # plt.show()
    # 생성 결과물 확인
    z = torch.randn(BATCH_SIZE, 128).to(DEVICE)
    fake_images = G(z)
    for i in range(50):
        fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (224, 224))
        # plt.imshow(fake_images_img)
        plt.imshow(fake_images_img, cmap='gray')
        plt.savefig(f'/opt/ml/v2/output/gan99/epoch_{epoch}_test_gan{i+2}.png')

    #######################################################

    # # -- model
    # model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model_module(
    #     num_classes=num_classes
    # ).to(device)
    # model = torch.nn.DataParallel(model)
    #
    # # -- loss & metric
    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    # opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = opt_module(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=5e-4
    # )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # # -- logging
    # logger = SummaryWriter(log_dir=save_dir)
    # with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    #     json.dump(vars(args), f, ensure_ascii=False, indent=4)
    #
    # best_val_acc = 0
    # best_val_loss = np.inf
    # for epoch in range(args.epochs):
    #     # train loop
    #     model.train()
    #     loss_value = 0
    #     matches = 0
    #     for idx, train_batch in enumerate(train_loader):
    #         inputs, labels = train_batch
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         outs = model(inputs)
    #         preds = torch.argmax(outs, dim=-1)
    #         loss = criterion(outs, labels)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         loss_value += loss.item()
    #         matches += (preds == labels).sum().item()
    #         if (idx + 1) % args.log_interval == 0:
    #             train_loss = loss_value / args.log_interval
    #             train_acc = matches / args.batch_size / args.log_interval
    #             current_lr = get_lr(optimizer)
    #             print(
    #                 f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
    #                 f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
    #             )
    #             logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
    #             logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
    #
    #             loss_value = 0
    #             matches = 0
    #
    #     scheduler.step()
    #
    #     # val loop
    #     with torch.no_grad():
    #         print("Calculating validation results...")
    #         model.eval()
    #         val_loss_items = []
    #         val_acc_items = []
    #         figure = None
    #         for val_batch in val_loader:
    #             inputs, labels = val_batch
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #
    #             outs = model(inputs)
    #             preds = torch.argmax(outs, dim=-1)
    #
    #             loss_item = criterion(outs, labels).item()
    #             acc_item = (labels == preds).sum().item()
    #             val_loss_items.append(loss_item)
    #             val_acc_items.append(acc_item)
    #
    #             if figure is None:
    #                 inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
    #                 inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
    #                 figure = grid_image(
    #                     inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
    #                 )
    #
    #         val_loss = np.sum(val_loss_items) / len(val_loader)
    #         val_acc = np.sum(val_acc_items) / len(val_set)
    #         best_val_loss = min(best_val_loss, val_loss)
    #         if val_acc > best_val_acc:
    #             print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
    #             torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
    #             best_val_acc = val_acc
    #         torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
    #         print(
    #             f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
    #             f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
    #         )
    #         logger.add_scalar("Val/loss", val_loss, epoch)
    #         logger.add_scalar("Val/accuracy", val_acc, epoch)
    #         logger.add_figure("results", figure, epoch)
    #         print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224,224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet18', help='model type (default: ResNet18)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=4, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
