import torch.nn as nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Accuracy:
    def __init__(self, topk=1):
        self.name = f"acc{topk}"
        self.topk = topk

    def __call__(self, output, target):
        """Args:
            output (Tensor): raw logits of shape (N, C)
            target (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)"""
        if len(target.shape) == 2:
            target = target.argmax(1)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[: self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul(100.0 / output.size(0))


class AccuracyNN(nn.Module):
    def __init__(self, topk=1):
        super().__init__()
        self.name = f"accNN{topk}"
        self.topk = topk

    def forward(self, output, target):
        """Args:
            output (Tensor): raw logits of shape (N, C)
            target (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C)"""
        if len(target.shape) == 2:
            target = target.argmax(1)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[: self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul(100.0 / output.size(0))


class Hits(nn.Module):
    def __init__(self, topk=1):
        super().__init__()
        self.name = f"hits{topk}"
        self.topk = topk

    def forward(self, output, target):
        if len(target.shape) == 2:
            target = target.argmax(1)
        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[: self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k
