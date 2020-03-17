from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class Mode(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Loss(_Loss):
    """Loss which supports addition and multiplication"""

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return WeightedLoss(self, value)
        else:
            raise ValueError("Loss should be multiplied by int or float")

    def __rmul__(self, other):
        return self.__mul__(other)


class WeightedLoss(Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = torch.Tensor([weight])

    def forward(self, *inputs):
        l = self.loss(*inputs)
        self.weight = self.weight.to(l.device)
        return l * self.weight[0]


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)


class CrossEntropyLoss(Loss):
    """
    CE with optional smoothing and support for multiple positive labels.
    Can accept one-hot encoded y_trues
    Supports only one reduction for now
    """

    def __init__(self, mode="multiclass", smoothing=0.0):
        """
        Args:
            mode (str): Metric mode {'binary', 'multiclass'}
                'binary' - calculate binary cross entropy
                'multiclass' - calculate categorical cross entropy
            smoothing (float): How much to smooth values toward uniform
        """
        super().__init__()
        self.mode = Mode(mode)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.name = "BCE"

    def forward(self, y_pred, y_true):
        if self.mode == Mode.BINARY:
            y_pred, y_true = y_pred.squeeze(), y_true.squeeze()
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
            return loss

        if len(y_true.shape) != 1:
            y_true_one_hot = y_true.float()
        else:
            num_classes = y_pred.size(1)
            y_true_one_hot = torch.zeros(y_true.size(0), num_classes, dtype=torch.float, device=y_pred.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)
        y_pred = y_pred.float()
        logprobs = F.log_softmax(y_pred, dim=1)
        # multiple labels handling
        nll_loss = -logprobs * y_true_one_hot
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    if hasattr(x, "dataset"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(
    inputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
    smooth_dist=None,
    from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLossSmooth(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean", smooth_eps=None, smooth_dist=None, from_logits=True
    ):
        super(CrossEntropyLossSmooth, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )


class MultiLabelSoftMarginLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, y_pred, y_true):
        if len(y_true.shape) != 1:
            y_true_one_hot = y_true.float()
        else:
            num_classes = y_pred.size(1)
            y_true_one_hot = torch.zeros(y_true.size(0), num_classes, dtype=torch.float, device=y_pred.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)

        return self.loss(y_pred, y_true_one_hot)
