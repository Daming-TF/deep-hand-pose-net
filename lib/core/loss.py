import math
import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        # 使用均方损失函数 l2求和取平均值
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(
            1, 1
        )  # dict(21) - (batch, h*w)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # Q1：use_target_weight是啥，目标权重是指？
            # Q2：loss均方误差计算完为啥要乘0.5？
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx]),
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        # Q3：上面criterion已经取平均为啥这里也要取平均，这种策略的原因是啥？
        return loss / num_joints


class WingLoss(nn.Module):
    """Wing Loss. paper ref: 'Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks'
    Feng et al. CVPR'2018.
    Args:
        omega (float):              Also referred to as width.
        epsilon (float):            Also referred to as curvature.
        use_target_weight (bool):   Option to use weighted MSE loss.
                                    Different joint types may have different target weights.
        loss_weight (float):        Weight of the loss. Default: 1.0.
    """

    def __init__(
        self, omega=10.0, epsilon=2.0, use_target_weight=False, loss_weight=1.0
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # constant that smoothly links the piecewise-defined linear and nonlinear parts
        self.C = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))

    def criterion(self, pred, target):
        """Criterion of wingloss.
        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)
        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon),
            delta - self.C,
        )

        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.
        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)
        Args:
            output (torch.Tensor[N, K, D]):         Output regression.
            target (torch.Tensor[N, K, D]):         Target regression.
            target_weight (torch.Tensor[N,K,1]):    Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            if output.ndim == 4:
                output = output.view(output.size(0), output.size(1), -1)
                target = target.view(target.size(0), target.size(1), -1)

            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight
