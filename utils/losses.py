from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.  Options are'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # https://github.com/megvii-model/HINet/blob/main/basicsr/models/losses/loss_util.py

    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        return loss

    return loss

def cross_entropy(inputs: torch.Tensor, 
                  targets: torch.Tensor,
                  weight: Optional[torch.Tensor] = None,
                  class_weight: Optional[torch.Tensor] = None,
                  reduction="mean",
                  ignore_index: int=-100)->torch.Tensor:
    """Calculate the CrossEntropy Loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # https://mmdetection.readthedocs.io/en/v2.16.0/_modules/mmdet/models/losses/cross_entropy_loss.html#cross_entropy
    loss = F.cross_entropy(inputs, targets, weight=class_weight, reduction="none", ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction)

    return loss

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "none",
        eps: float = 1e-6,
) -> torch.Tensor:
    # https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/dice_loss.py
    inputs = F.softmax(inputs, dim=1)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    if inputs.shape != targets.shape:
        raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")

    # flatten prediction and label tensors
    inputs = inputs.flatten(1)
    targets = targets.flatten(1).float()    

    intersection = torch.sum(inputs * targets, 1)
    denominator = torch.sum(inputs, 1) + torch.sum(targets, 1)

    # calculate the dice loss
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1 - dice_score

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(inputs)
    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss 

def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
) -> torch.Tensor:
    
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Sigmoid activation on inputs
    probs = torch.sigmoid(inputs)

    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    inputs = inputs.float()
    targets = targets.float()

    if inputs.shape != targets.shape:
        raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")
    pt = (1 - probs) * targets + probs * (1 - targets)

    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)

    loss = focal_weight * F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if weight is not None:
        assert weight.dim() == 1, f"Weight dimension must be `weight.dim()=1`, current dimension {weight.dim()}"
        weight = weight.float()
        if inputs.dim() > 1:
            weight = weight.view(-1, 1)

    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss

def silog_loss(pred: torch.Tensor,
               target: torch.Tensor,
               weight: Optional[torch.Tensor] = None,
               eps: float = 1e-4,
               reduction: Union[str, None] = 'mean') -> torch.Tensor:
    
    """Computes the Scale-Invariant Logarithmic (SI-Log) loss between
    prediction and target.

    Args:
        pred (Tensor): Predicted output.
        target (Tensor): Ground truth.
        weight (Optional[Tensor]): Optional weight to apply on the loss.
        eps (float): Epsilon value to avoid division and log(0).
        reduction (Union[str, None]): Specifies the reduction to apply to the
            output: 'mean', 'sum' or None.
        avg_factor (Optional[int]): Optional average factor for the loss.

    Returns:
        Tensor: The calculated SI-Log loss.
    """
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2)
    
    pred = pred.flatten(1)
    target = target.flatten(1).float()   
    
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(pred.clamp(min=eps))

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(
        dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight, reduction)
    return loss


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(
            self,
            class_weights: Optional[torch.Tensor] = None,
            reduction: str = "mean",
            loss_weight: float = 1.0,
    )->torch.Tensor:
        super().__init__()
        self.class_weight = class_weights
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
    ):
        loss = self.loss_weight * cross_entropy(
            inputs, targets, weight, class_weight=self.class_weight, reduction=self.reduction, ignore_index=ignore_index
        )

        return loss

class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(
            self,
            reduction: str = "mean",
            loss_weight: Optional[float] = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * dice_loss(inputs, targets, weight=weight, reduction=self.reduction, eps=self.eps)

        return loss

class DiceCELoss(nn.Module):
    """Dice Cross Entropy Loss"""
    def __init__(
            self,
            reduction: str = "mean",
            dice_weight: float = 1.0,
            ce_weight: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None):
        # dice loss
        dice = dice_loss(inputs, targets, weight=weight, reduction=self.reduction, eps=self.eps)
        # entropy loss
        ce = cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)
        # accumulate loss according to given weights
        loss = self.dice_weight * dice + ce * self.ce_weight

        return loss

class FocalLoss(nn.Module):
    """Sigmoid Focal Loss"""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * sigmoid_focal_loss(inputs, targets, weight, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction)

        return loss

class SiLogLoss(nn.Module):
    """Compute SiLog loss.

    Args:
        reduction (str, optional): The method used
            to reduce the loss. Options are "none",
            "mean" and "sum". Defaults to 'mean'.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        eps (float): Avoid dividing by zero. Defaults to 1e-3.
        loss_name (str, optional): Name of the loss item. If you want this
            loss item to be included into the backward graph, `loss_` must
            be the prefix of the name. Defaults to 'loss_silog'.
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_silog'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None  
    ):
       
        loss = self.loss_weight * silog_loss(
            inputs,
            targets,
            weight,
            eps=self.eps,
            reduction=self.reduction           
        )

        return loss