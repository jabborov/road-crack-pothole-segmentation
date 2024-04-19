"""Evaluation Metric
This script claculates metrics to evaluate UNet model performance.
"""

import numpy as np
import torch

def pix_acc(outputs: torch.Tensor, targets: torch.Tensor, batch_size: int):
    """Pixel Accuracy

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
    """
    acc = 0.0
    for idx in range(batch_size):
        output = outputs[idx]
        target = targets[idx]
        correct = torch.sum(torch.eq(output, target).long())
        acc += correct / np.prod(np.array(output.shape)) / batch_size
    return acc.item()

def iou(outputs: torch.Tensor, targets: torch.Tensor, batch_size: int, num_classes: int):
    """Intersection Over Union

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
        n_classes (int): number of segmentation classes
    """
    eps = 1e-6
    class_iou = np.zeros(num_classes)
    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()
        targets_cpu = targets[idx].cpu()

        for c in range(num_classes):
            i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
            i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
            intersection = np.intersect1d(i_outputs, i_targets).size
            union = np.union1d(i_outputs, i_targets).size
            class_iou[c] += (intersection + eps) / (union + eps)

    class_iou /= batch_size

    return class_iou

