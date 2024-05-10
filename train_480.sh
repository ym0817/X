nohup python tools/train.py \
-f exps/default/yolox_nano.py \
-n yolox_nano \
-d 0 \
-b 60 \
--fp16 \
-o \
--resume \
--start_epoch=10 \
-c /mnt2T/YM/Train/x/YOLOX_outputs/yolox_nano/latest_ckpt.pth \
> nohup.out 2>&1 &
tail -f nohup.out

# --resume True \
#-c yolox_tiny.pth
 # [--cache]
 # python setup.py develop