import logging
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import namedtuple
from pathlib import Path

from lib.utils.files import get_name


def create_logger(cfg, cfg_name, phase="train"):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    # set up logger
    if not root_output_dir.exists():
        print("=> creating {}".format(root_output_dir))
        root_output_dir.mkdir()

    if cfg.DATASET.HYBRID_JOINTS_TYPE:
        dataset = cfg.DATASET.DATASET + "_" + cfg.DATASET.HYBRID_JOINTS_TYPE
    else:
        dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(":", "_")

    model = cfg.MODEL.NAME
    cfg_name = get_name(cfg_name)

    final_output_dir = root_output_dir / dataset / model / cfg_name
    print("=> creating {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_file = "{}_{}_{}.log".format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    tensorboard_log_dir = (
        Path(cfg.LOG_DIR) / dataset / model / (cfg_name + "_" + time_str)
    )
    print("=> creating {}".format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    summary = []

    ModuleDetails = namedtuple(
        "Layer",
        ["name", "input_size", "output_size", "num_parameters", "multiply_adds"],
    )
    hooks = []
    layer_instances = {}

    def add_hooks(module):
        def hook_fn(module_, input_data, output):
            class_name = str(module_.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0
            if (
                class_name.find("Conv") != -1
                or class_name.find("BatchNorm") != -1
                or class_name.find("Linear") != -1
            ):
                # 遍历模块参数
                for param_ in module_.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            # hasattr判断对象是否包含对应的属性
            if class_name.find("Conv") != -1 and hasattr(module_, "weight"):
                flops = (
                    torch.prod(torch.LongTensor(list(module_.weight.data.size())))
                    * torch.prod(torch.LongTensor(list(output.size())[2:]))
                ).item()
            elif isinstance(module_, nn.Linear):
                flops = (
                    torch.prod(torch.LongTensor(list(output.size())))
                    * input_data[0].size(1)
                ).item()

            if isinstance(input_data[0], list):
                input_data = input_data[0]
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input_data[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops,
                )
            )

        if (
            not isinstance(module, nn.ModuleList)
            and not isinstance(module, nn.Sequential)
            and module != model
        ):
            # register_forward_hook（）：
            # 作用：在module上注册一个forward hook。 每次调用forward()计算输出的时候，这个hook就会被调用。
            hooks.append(module.register_forward_hook(hook_fn))

    # eval()作用：
    # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，
    # 而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    model.eval()
    # apply(fn)功能————实际使用的是深度优先算法
    # 将一个函数fn递归地应用到模块自身以及该模块的每一个子模块中(即在函数.children()中返回的子模块)。该方法通常用来初始化一个模型的参数
    model.apply(add_hooks)

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    space_len = item_length
    details = ""
    if verbose:
        details = (
            "Model Summary"
            + os.linesep
            + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                " " * (space_len - len("Name")),
                " " * (space_len - len("Input Size")),
                " " * (space_len - len("Output Size")),
                " " * (space_len - len("Parameters")),
                " " * (space_len - len("Multiply Adds (Flops)")),
            )
            + os.linesep
            + "-" * space_len * 5
            + os.linesep
        )

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += (
                "{}{}{}{}{}{}{}{}{}{}".format(
                    layer.name,
                    " " * (space_len - len(layer.name)),
                    layer.input_size,
                    " " * (space_len - len(str(layer.input_size))),
                    layer.output_size,
                    " " * (space_len - len(str(layer.output_size))),
                    layer.num_parameters,
                    " " * (space_len - len(str(layer.num_parameters))),
                    layer.multiply_adds,
                    " " * (space_len - len(str(layer.multiply_adds))),
                )
                + os.linesep
                + "-" * space_len * 5
                + os.linesep
            )

    details += (
        os.linesep
        + "Total Parameters: {:,}".format(params_sum)
        + os.linesep
        + "-" * space_len * 5
        + os.linesep
    )
    details += (
        "Total Multiply Adds (For Convolution and Linear Layers only): {:,} MFLOPs".format(
            flops_sum / (1024 ** 2)
        )
        + os.linesep
        + "-" * space_len * 5
        + os.linesep
    )
    details += "Number of Layers" + os.linesep

    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def set_cudnn_related_setting(cfg):
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    cudnn.benchmark = cfg.CUDNN.BENCHMARK   # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


def copy_model_cfg_to_file(cfg, args, final_output_dir):
    # 获取当前工程的绝对路径
    this_dir = os.path.dirname(os.path.dirname(__file__))  # path of lib
    shutil.copy2(
        os.path.join(this_dir, "models", cfg.MODEL.NAME + ".py"), final_output_dir
    )
    shutil.copy2(os.path.join(this_dir, "../", args.cfg), final_output_dir)  # path of p
