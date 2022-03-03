

python train.py --model modified_efficientnet-b4 \
                            --img_resize 412 \
                            --img_crop 380 \
                            --train_batch_size 32 \
                            --loss LDAM \
                            --epochs 8 \
                            --mode inference
#
python train.py --model modified_efficientnet-b4 \
                            --img_resize 412 \
                            --img_crop 380 \
                            --train_batch_size 32 \
                            --loss focal \
                            --epochs 8 \
                            --mode inference
