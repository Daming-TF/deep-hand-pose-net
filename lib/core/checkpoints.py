import os
import torch


def load_model_to_continue(checkpoint_path, model, optimizer, logger):
    logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
    # torcj.load('**.pt'): 默认加载方式，使用cpu加载cpu训练得出的模型或者用gpu调用gpu训练的模型：
    # 参数反序列化为python dict
    checkpoint = torch.load(checkpoint_path)
    begin_epoch = checkpoint["epoch"]
    best_perf = checkpoint["perf"]
    last_epoch = checkpoint["epoch"]
    # # 加载训练好的参数
    model.load_state_dict(checkpoint["state_dict"])
    # 将目前optimizer中的params参数填充到statedict中
    # 然后用statedict中的state和params_group替换掉目前optimizer中的state和param_group
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info(
        "=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_path, checkpoint["epoch"]
        )
    )
    return begin_epoch, best_perf, last_epoch


def save_checkpoint(states, is_best=False, output_dir="", filename="checkpoint.pth"):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and "state_dict" in states:
        torch.save(states["best_state_dict"], os.path.join(output_dir, filename))
