#python train.py --model modified_efficientnet-b4 \
#                            --img_resize 412 \
#                            --img_crop 380 \
#                            --train_batch_size 32 \
#                            --epochs 4 \
#                            --mode train

python k_fold.py --model modified_efficientnet-b3 \
                            --img_resize 332 \
                            --img_crop 300 \
                            --train_batch_size 64 \
                            --epochs 13 \
                            --mode train