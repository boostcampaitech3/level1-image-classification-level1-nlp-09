#!/bin/bash
python train.py --model modified_efficientnet-b3 \
                --img_resize 332 \
                --img_crop 300 \
                --epochs 15 \
                --train_batch_size 64

## 다른 모델 예시
#python model_train_split.py --model efficientnet-b0 \
#                            --img_resize 256 \
#                            --img_crop 224
#
#python model_train_split.py --model efficientnet-b1 \
#                            --img_resize 272 \
#                            --img_crop 240
#
#python model_train_split.py --model efficientnet-b2 \
#                            --img_resize 292 \
#                            --img_crop 260
#
#python model_train_split.py --model efficientnet-b3 \
#                            --img_resize 332 \
#                            --img_crop 300
#
#python model_train_split.py --model modified_efficientnet-b4 \
#                            --img_resize 412 \
#                            --img_crop 380 \
#                            --train_batch_size 32
