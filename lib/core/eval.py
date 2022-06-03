import logging
import os
import torch
import numpy as np

from lib.core.inference import (
    get_max_preds,
    get_final_preds,
    get_final_preds_from_regressor,
    get_final_preds_align,
)
from lib.utils.vis import save_debug_images


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def dist_acc(dists, thr=0.5):
    # Return percentage below threshold while ignoring values with a -1
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))  # (21, N)

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def accuracy(output, target, size=None, hm_type="gaussian", thr=0.1):
    """
    Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations/
    First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    """
    idx = list(range(output.shape[1]))      # 热图数量
    norm = 1.0
    pred = None
    if hm_type == "gaussian":
        if output.ndim == 4:  # (N, 21, h, w)
            pred, _ = get_max_preds(output)  # (N, 21, 2)
            target, _ = get_max_preds(target)  # (N, 21, 2)
            h = output.shape[2]
            w = output.shape[3]
            # ref:https://github.com/HowieMa/NSRMhand/blob/master/src/utils.py#L77
            norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10 * (2 / 1.5)
        elif output.ndim == 3:  # (N, 21, 2)
            pred = (output + 0.5) * size[0]  # [-0.5, 0.5] -> [0., 1.] * size
            target = (target + 0.5) * size[0]  # [-0.5, 0.5] -> [0., 1.] * size
            norm = np.ones((pred.shape[0], 2)) * np.array(size) / 1.5
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred


def validate_align(
    config, val_loader, val_dataset, model, criterion, output_dir, writer_dict=None
):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    with torch.no_grad():
        for i, (input_data, target, target_weight, meta, target_reg) in enumerate(
            val_loader
        ):
            outputs = model(input_data)

            target = target.cuda(non_blocking=True)
            target_reg = target_reg.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if config.LOSS.TYPE == "heatmap":
                loss = criterion(outputs, target, target_weight)
            else:
                loss = criterion(outputs, target_reg, target_weight)

            # measure accuracy and record loss
            num_images = input_data.size(0)
            losses.update(loss.item(), num_images)

            if config.LOSS.TYPE == "heatmap":
                _, avg_acc, cnt, _ = accuracy(
                    outputs.detach().cpu().numpy().copy(),
                    target.detach().cpu().numpy().copy(),
                )
            else:
                _, avg_acc, cnt, _ = accuracy(
                    outputs.detach().cpu().numpy().copy(),
                    target_reg.detach().cpu().numpy().copy(),
                    size=config.MODEL.IMAGE_SIZE,
                )
            acc.update(avg_acc, cnt)

            c = meta["center"].numpy()
            s = meta["scale"].numpy()
            score = meta["score"].numpy()
            warp_matrix = meta["warp_matrix"].numpy()

            if config.LOSS.TYPE == "heatmap":
                # preds, maxvals = get_final_preds(config, outputs.clone().cpu().numpy().copy(), c, s)
                preds, maxvals = get_final_preds_align(
                    config, outputs.clone().cpu().numpy().copy(), c, s, warp_matrix
                )
            else:
                preds, maxvals = get_final_preds_from_regressor(
                    config, outputs.clone().cpu().numpy().copy(), c, s
                )

            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta["image"])
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = f"Test: [{i}/{len(val_loader)}] - Loss: {losses.avg:.4f} Accuracy: {acc.avg:.3f}"
                logger.info(msg)

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'test'), i)
            # if config.LOSS.TYPE == "heatmap":
            #     save_debug_images(config, input_data, meta, target, preds, outputs, prefix)
            # else:
            #     save_debug_images(config, input_data, meta, target_reg, preds, outputs, prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums
        )

        model_name = config.MODEL.NAME
        _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict["writer"]
            global_steps = writer_dict["valid_global_steps"]
            writer.add_scalar("valid_loss", losses.avg, global_steps)
            writer.add_scalar("valid_acc", acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars("valid", dict(name_value), global_steps)
            else:
                writer.add_scalars("valid", dict(name_values), global_steps)
            writer_dict["valid_global_steps"] = global_steps + 1

    return perf_indicator


def validate(
    config, val_loader, val_dataset, model, criterion, output_dir, writer_dict=None
):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    # 进入测试模式：不需要计算梯度，也不会进行反向传播
    with torch.no_grad():
        for i, (input_data, target, target_weight, meta, target_reg) in enumerate(
            val_loader
        ):
            outputs = model(input_data)

            target = target.cuda(non_blocking=True)
            target_reg = target_reg.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if config.LOSS.TYPE == "heatmap":
                loss = criterion(outputs, target, target_weight)
            else:
                loss = criterion(outputs, target_reg, target_weight)

            # measure accuracy and record loss
            num_images = input_data.size(0)
            losses.update(loss.item(), num_images)

            if config.LOSS.TYPE == "heatmap":
                _, avg_acc, cnt, _ = accuracy(
                    outputs.detach().cpu().numpy().copy(),
                    target.detach().cpu().numpy().copy(),
                )
            else:
                _, avg_acc, cnt, _ = accuracy(
                    outputs.detach().cpu().numpy().copy(),
                    target_reg.detach().cpu().numpy().copy(),
                    size=config.MODEL.IMAGE_SIZE,
                )
            acc.update(avg_acc, cnt)

            c = meta["center"].numpy()
            s = meta["scale"].numpy()
            score = meta["score"].numpy()

            if config.LOSS.TYPE == "heatmap":
                preds, maxvals = get_final_preds(
                    config, outputs.clone().cpu().numpy().copy(), c, s
                )
            else:
                preds, maxvals = get_final_preds_from_regressor(
                    config, outputs.clone().cpu().numpy().copy(), c, s
                )

            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta["image"])
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = f"Test: [{i}/{len(val_loader)}] - Loss: {losses.avg:.4f} Accuracy: {acc.avg:.3f}"
                logger.info(msg)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums
        )

        model_name = config.MODEL.NAME
        _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict["writer"]
            global_steps = writer_dict["valid_global_steps"]
            writer.add_scalar("valid_loss", losses.avg, global_steps)
            writer.add_scalar("valid_acc", acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars("valid", dict(name_value), global_steps)
            else:
                writer.add_scalars("valid", dict(name_values), global_steps)
            writer_dict["valid_global_steps"] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|-----" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."
    logger.info(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )
