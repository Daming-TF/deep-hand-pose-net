import os
import pprint
import torch
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# noinspection PyUnresolvedReferences
import _init_path
import lib.models
import lib.dataset
from lib.config import cfg, parse_args, update_config
from lib.core.checkpoints import load_model_to_continue, save_checkpoint
from lib.core.eval import validate
from lib.core.loss import JointsMSELoss, WingLoss
from lib.core.optimizer import get_optimizer
from lib.core.train import train
from lib.utils import (
    create_logger,
    set_cudnn_related_setting,
    copy_model_cfg_to_file,
    get_model_summary,
)


def main():
    args = parse_args()
    update_config(cfg, args)

    # final_output_dir格式类似于'output/hand_coco/mobilenetv2/debug'
    # tb_log_dir格式类似于'log/hand_coco/mobilenetv2/debug_2022-04-01-15-21'
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "train")
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    model = getattr(lib.models, cfg.MODEL.NAME).get_pose_net(cfg, is_train=True)
    # logger.info(pprint.pformat(model))

    set_cudnn_related_setting(cfg)  # cudnn related setting
    copy_model_cfg_to_file(cfg, args, final_output_dir)  # copy model file

    # TensorboardX可以提供中很多的可视化方式，SummaryWriter就是定义一个这样的实例，其中log_dir表示生成文件所在的目录
    # 到时候在定义实例输入的路径同级目录下使用命令行，例如：
    # python -m tensorboard.main --logdir=E:\hand_project2\deep-hand-pose-net\log
    # 即可可视化相关信息，比如loss
    # 若相比较两个模型，可输入参数--logdir_spec，例如：
    # tensorboard --logdir_spec SmallNano-0:{地址1},Nano-0:{地址2},SmallNano-2:{地址3} --port=8887 --host 0.0.0.0
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # calculate model params. and MFlops info
    # 计算相关指标：Total Parameters，Multiply Adds
    result = get_model_summary(
        model, torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    )
    logger.info(result)

    # # 用多个GPU来加速训练  device_ids表示为GPU代号,如[0, 1, 2]表示三块GPU
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # define loss function (criterion)
    if cfg.LOSS.TYPE == "heatmap":
        criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    elif cfg.LOSS.TYPE == "regressor":
        criterion = WingLoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    else:
        raise Exception(" [!] Loss type shoulde one of the heatmap and regressor")

    # Data loading code
    # torchvision.transforms是pytorch中的图像预处理包，一般用Compose把多个步骤整合到一起
    # ToTensor 目的是将输入的数据shape W，H，C ——> C，H，W 并将每个channel范围转换为[0, 1]
    # Normalize 目的是将每个通道的数值归一化，使数值符合正态分布
    # # mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]表示的是从coco数据集中随机抽样计算得到的
    train_dataset = getattr(lib.dataset, cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,           # '/workspace/cpfs-data/Data/debug'
        cfg.DATASET.TRAIN_SET,      # 'train2017'
        True,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]       # mean,std的计算可以参考https://zhuanlan.zhihu.com/p/414242338
                ),
            ]
        ),
    )
    valid_dataset = getattr(lib.dataset, cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.VAL_SET,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    """
        PyTorch为什么要dataset和dataloader分开？
            更加方便的解耦，dataset的存在旨在给所有的数据集一个统一的结构(使用中括号取数据)
            对于每一个数据集，读取方法几乎都是不同的。
            而dataloader的行为大都趋于一致，按照batch取数据，最多是在如何对数据采样等方法上有一些定制。
            因此将二者分开，用户只需要重写经常变化的部分(dataset)，对于dataloader保持稳定性，毕竟不常扩展。
        """
    # batch_size：即一次训练所抓取的数据样本数量
    # shuffle 在每次迭代训练时是否将数据洗牌 ，默认设置是False
    # pin_memory 内存寄存，默认为False； 在数据返回前，是否将数据复制到CUDA内存中。在cfg默认设置为True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = 0.0
    # best_model = None
    last_epoch = -1
    # 设置参数梯度优化器，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    # 一般通过step()方法来对所有的参数进行更新
    # 设置为Adam算法，与sgd不同，他的学习率是根据迭代次数慢慢变小的（三个影响参数，一阶二阶矩阵估计衰减率以及学习率）
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    # 如果checkpoints.pth路径存在则会根据上一次训练结果继续训练
    checkpoint_path = os.path.join(final_output_dir, "checkpoint.pth")
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_path):
        begin_epoch, best_perf, last_epoch = load_model_to_continue(
            checkpoint_path, model, optimizer, logger
        )

    # 在模型训练的优化部分，调整最多的一个参数就是学习率，合理的学习率可以使优化器快速收敛。 一般在训练初期给予较大的学习率，随着训练的进行，学习率逐渐减小
    # lr_scheduler.StepLR: 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。间隔单位是step
    # lr_scheduler.MultiStepLR: 按设定的间隔调整学习率
    # lr_scheduler.ExponentialLR: 按指数衰减调整学习率
    #lr_scheduler.CosineAnnealingLR：以余弦函数为周期，并在每个周期最大值时重新设置学习率
    # 按设定的间隔调整学习率 cfg.TRAIN.LR_STEP： [360,380,400]，
    # 当epoch次数到达360， 380， 400则lr×gamma（gamma对应下面cfg.TRAIN.LR_FACTO：0.1）
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train(
            cfg,
            train_loader,
            model,
            criterion,      #
            optimizer,
            epoch,
            final_output_dir,
            tb_log_dir,
            writer_dict,
        )

        # 如何知道epoch是多少？
        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(
            cfg,
            valid_loader,
            valid_dataset,
            model,
            criterion,
            final_output_dir,
            writer_dict,
        )
        states = {
            "epoch": epoch + 1,
            "model": cfg.MODEL.NAME,
            "state_dict": model.state_dict(),
            "best_state_dict": model.module.state_dict(),
            "perf": perf_indicator,
            "optimizer": optimizer.state_dict(),
        }

        logger.info("=> saving checkpoint to {}".format(final_output_dir))
        if perf_indicator > best_perf:
            best_perf = perf_indicator
            save_checkpoint(
                states=states,
                is_best=True,
                output_dir=final_output_dir,
                filename="model_best.pth",
            )
        else:
            save_checkpoint(
                states=states,
                is_best=False,
                output_dir=final_output_dir,
                filename="checkpoint.pth",
            )

    logger.info(
        f"=> saving final model state to {os.path.join(final_output_dir, 'final_state.pth')}"
    )
    save_checkpoint(
        states=model.module.state_dict(),
        output_dir=final_output_dir,
        filename="final_state.pth",
    )
    writer_dict["writer"].close()


if __name__ == "__main__":
    main()
