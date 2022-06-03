import torch.optim as optim


def get_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV,
        )
    elif cfg.TRAIN.OPTIMIZER == "adam":
        # nn.Module.parameters可以遍历模型参数
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)     # lr：0.001
    else:
        raise Exception(f" [!] Optimizer {cfg.TRAIN.OPTIMIZER} is not considered!")

    return optimizer
