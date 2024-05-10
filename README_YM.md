1 从头训练
python tools/train.py -f exps/example/yolox_voc/yolox_nano.py -c models/yolox_s.pth -d 0 -b 32 --fp16



2 恢复训练
python tools/train.py -f exps/default/yolox_nano.py -n yolox_nano \
-d 0 -b 60 --fp16 -o --resume --start_epoch=10 -c /mnt2T/YM/Train/x/YOLOX_outputs/yolox_nano/latest_ckpt.pth

python tools/train.py -f exps/example/yolox_voc/yolox_voc_nano.py -c models/yolox_nano.pth -d 2 -b 32 --fp16




3 评估命令
python tools/eval.py -f exps/example/yolox_voc/yolox_voc_s.py -b 32 -d 0 --conf 0.4 -- fp16

python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                               yolox-m
                               yolox-l
                               yolox-x

To reproduce speed test, we use the following command:
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 1 -d 0 --conf 0.001 --fp16 --fuse
                               yolox-m
                               yolox-l
                               yolox-x


4 demo命令
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py --path datasets/MOT17/images/test/MOT17-12-SDP/img1 --conf 0.5 --tsize 640 --save_result

python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]

python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]

python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]




5 导出onnx
python tools/export_onnx.py --output-name yolox_tiny.onnx -f exps/default/yolox_tiny.py -n yolox_tiny  -c yolox_tiny.pth

