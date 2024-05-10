参考： https://zhuanlan.zhihu.com/p/425375252?utm_id=0


一、 训练过程
训练命令
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o --cach

2. 训练过程

训练的代码在train.py 中，首先对训练的参数进行解析， 具体参数如下表所示，然后调用get_exp(args.exp_file, args.name) 获取exp，调用exp.merge(args.opts)并merge训练参数，然后启动训练， 调用

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)   # 获取模型文件
    exp.merge(args.opts)       # 和并参数， 针对有的参数进行和并

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )


训练的参数如下：

参数名称	描述	default
"-expn", "--experiment-name"		None
"-n", "--name"	指定训练的模型名称	
--dist-backend	distributed backend	nccl
--dist-url	url used to set up distributed training	None
"-b", "--batch-size"	batch size	64
"-d", "--devices"	device for training	None
"-f", "--exp_file",	plz input your experiment description file	None
--resume	resume training	False
"-c", "--ckpt"	checkpoint file	None
"-e", "--start_epoch"	resume training start epoch	None
--num_machines	num of node for training	1
--machine_rank	node rank for multi-node training	0
--fp16	Adopting mix precision training.	False
--cache	Caching imgs to RAM for fast training.	False
"-o", "--occupy"	"Modify config options using the command-line"	argparse.REMAINDER




2.1 模型获取

通过传递模型名字获取具体的模型，模型的名字比如yolox-s， 最终调用的是get_exp_by_file， 其首先把各种模型文件的目录加入到python的系统路径搜索里面，在添加根据模型的名字导入各个模块，其中每个模块都是继承MyExp,然后调用每个模型的Exp函数。针对不同的模型，可以设置depth和width的值。

其中current_exp 是在yolox_base.py文件中定义的Exp类



exp = get_exp(args.exp_file, args.name)

def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))  # 添加具体包含各种模型文件的路径， 为exps/default
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])  # 导入各种module
        exp = current_exp.Exp()  #
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]




2.2 模型封装代码 Exp 类，该类封装了模型的配置，创建， 数据loader， 前处理， 后处理， 优化，学习率等， 在文件yolox_bashe.py中。





3. 模型训练

模型训练主要是分成beforetrain 、 traininepoch 和after_train 三部分；

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()
3.1 train前

主要是获取模型， 获取优化器，

def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()   # 获取模型
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        # 前5epoll 的学习率为0，后面的学习率为0.00015625 * batch_size, 为0.1  
        self.optimizer = self.exp.get_optimizer(self.args.batch_size) 

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    在这个函数中load模型的预训练值， 如果没有指定文件则不load参数
    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

学习率， 其中self.scheduler = "yoloxwarmcos"

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
3.2 train



3.3 train后



4. 数据加载器dataloader
train.py 文件

在trainer.py 函数"train_one_iter" 函数中调用prefetcher.next()

        def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
在函数 def before_train(self) 函数中 调用exp.get_data_loader获取 train_loader，并加 载数据预取DataPrefetcher

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)


针对VOC数据集， tainloader 的创建在文件yoloxvoc_s.py中

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            img_size=self.input_size,
            # 这个函数主要实现了
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=50,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=120,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader


VOCDetection dataset

该类主要实现VOC的数据集，主要实现如下两个函数， __len__函数返回数据的数据的个数

    def __len__(self):
        return len(self.ids)


另一个需要实现的函数getitem 返回一张图片的相关信息， 其中img为原图， target为检测框， img_info为图片的尺寸， index 为图像的索引；

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:   # 这个函数主要实现减去均值，色度条件，resize操作等，最终
           变成了RGB的图像，并交换了图像的channel，变成CHW的形式
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id  # 返回处理后的图像和label，该图像为CHW，RGB格式，尺寸为416X416




5. 自定义数据集





6. 操作命令



DEMO命令：
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py --path datasets/MOT17/images/test/MOT17-12-SDP/img1 --conf 0.25 --tsize 640 --save_result



训练命令
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -c models/yolox_s.pth -d 2 -b 32 --fp16



python tools/train.py -f exps/example/yolox_voc/yolox_voc_nano.py -c models/yolox_nano.pth -d 2 -b 32 --fp16

评估命令
python tools/eval.py -f exps/example/yolox_voc/yolox_voc_s.py -b 32 -d 2 --conf 0.4 -- fp16

demo命令
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py --path datasets/MOT17/images/test/MOT17-12-SDP/img1 --conf 0.5 --tsize 640 --save_result


