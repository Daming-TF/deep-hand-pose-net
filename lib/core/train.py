import logging
import os
import time
from lib.core.eval import AverageMeter, accuracy
from lib.utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    output_dir,
    tb_log_dir,
    writer_dict,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    # model.train()的作用是启用 Batch Normalization 和 Dropout
    model.train()

    end = time.time()
    for i, (input_data, target, target_weight, meta, target_reg) in enumerate(
        train_loader
    ):
        # measure data loading time
        data_time.update(time.time() - end)
        # .cuda()是为了将模型放在GPU上进行训练
        # non_blocking默认值为False, 通常我们会在加载数据时，将DataLoader的参数pin_memory设置为True,
        # DataLoader中参数pin_memory的作用是：
        # 将生成的Tensor数据存放在哪里，值为True时，意味着生成的Tensor数据存放在锁页内存中，这样内存中的Tensor转义到GPU的显存会更快
        target = target.cuda(non_blocking=True)
        target_reg = target_reg.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        outputs = model(input_data)
        if config.LOSS.TYPE == "heatmap":
            loss = criterion(outputs, target, target_weight)
        else:
            loss = criterion(outputs, target_reg, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()       # 将梯度归零
        """
        关于backward的使用：
        对标量作用：
        x = torch.ones(2,requires_grad=True)
        z = x + 2
        z.sum().backward()
        print(x.grad)
        
        ==》 tensor([1., 1.])
        
        对矩阵作用：
        z = torch.mm(x.view(1, 2), y)
        z.backward(torch.Tensor([[1., 1]]), retain_graph=True)
        print(f"x.grad: {x.grad}")
        print(f"y.grad: {y.grad}")
        
        ==》 x.grad：tensor([3., 7.])
        ==》 y.grad: tensor([[2., 2.],[1., 1.]])
        """
        # 注意：backward只能作用于标量
        loss.backward()             # 反向传播计算得到每个参数的梯度值
        optimizer.step()            # 通过梯度下降，更新参数

        # measure accuracy and record loss
        losses.update(loss.item(), input_data.size(0))      # input_data： tensor(batch_size, channel, width, height)

        # tensor.detach函数：返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        if config.LOSS.TYPE == "heatmap":
            _, avg_acc, cnt, pred = accuracy(
                outputs.detach().cpu().numpy().copy(),
                target.detach().cpu().numpy().copy(),
            )
        else:
            _, avg_acc, cnt, pred = accuracy(
                outputs.detach().cpu().numpy().copy(),
                target_reg.detach().cpu().numpy().copy(),
                size=config.MODEL.IMAGE_SIZE,
            )
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config == 0:
            speed = input_data.size(0) / batch_time.val
            msg = (
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t "
                f"Speed {speed:.1f} samples/s\t Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t "
                f"Loss {losses.val:.5f} ({losses.avg:.5f})\t Accuracy {acc.val:.3f} ({acc.avg:.3f})"
            )
            logger.info(msg)

            # update tensorboard info
            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_loss", losses.val, global_steps)
            writer.add_scalar("train_acc", acc.val, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

            prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
            if config.LOSS.TYPE == "heatmap":
                save_debug_images(
                    config, input_data, meta, target, pred * 4, outputs, prefix
                )
            else:
                save_debug_images(
                    config, input_data, meta, target_reg, pred, outputs, prefix
                )
