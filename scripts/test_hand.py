import os
import pprint
import torch
import torch.utils.data
import torchvision.transforms as transforms

# noinspection PyUnresolvedReferences
import _init_path
import lib.models
import lib.dataset
from lib.config import cfg, update_config, parse_args
from lib.core.eval import validate
from lib.core.loss import JointsMSELoss
from lib.utils import create_logger, set_cudnn_related_setting, get_model_summary


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "valid")
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    set_cudnn_related_setting(cfg)  # cudnn related setting
    model = getattr(lib.models, cfg.MODEL.NAME).get_pose_net(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, "model_best.pth")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file), strict=False)

    # calculate model params. and MFlops info
    result = get_model_summary(
        model, torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    )
    logger.info(result)
    model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    valid_dataset = getattr(lib.dataset, cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
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
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len((0,)),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir)


if __name__ == "__main__":
    main()
