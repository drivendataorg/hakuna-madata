import numpy as np
import torch
from torch.distributions import beta
from thunder_hammer.utils import to_python_float


def mixup(x, y, lam):
    index = torch.randperm(x.size(0))

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b


def mixup_pass_v1(net, criterion, inputs, targets, alpha, max_lambda=True):
    mixup_sampler = beta.Beta(alpha, alpha)

    lam = mixup_sampler.sample().to(inputs.device)
    if max_lambda:
        lam = torch.max(lam, 1 - lam)

    mixed_inputs, targets_a, targets_b = mixup(inputs, targets, lam)

    outputs = net(inputs)
    outputs_mixed = net(mixed_inputs)

    loss_orig = criterion(outputs, targets)
    loss_mixed = criterion(outputs_mixed, targets_b)

    loss = lam * loss_orig + (1 - lam) * loss_mixed
    return loss, outputs


def mixup_pass_v2(net, criterion, inputs, targets, alpha, max_lambda=True):
    mixup_sampler = beta.Beta(alpha, alpha)
    lam = mixup_sampler.sample().to(inputs.device)
    if max_lambda:
        lam = torch.max(lam, 1 - lam)

    inputs, targets_a, targets_b = mixup(inputs, targets, lam)

    outputs = net(inputs)

    loss_orig = criterion(outputs, targets)
    loss_mixed = criterion(outputs, targets_b)

    loss = lam * loss_orig + (1 - lam) * loss_mixed
    return loss, outputs


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # cut_rat = np.sqrt(1.0 - lam)
    cut_rat = np.sqrt(to_python_float(lam))
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(x, y, lam):
    index = torch.randperm(x.size(0))

    target_a = y
    target_b = y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def cutmix_pass(net, criterion, inputs, targets, alpha, max_lambda=True):
    mixup_sampler = beta.Beta(alpha, alpha)
    lam = mixup_sampler.sample().to(inputs.device)
    if max_lambda:
        lam = torch.min(lam, 1 - lam)

    inputs, targets_a, targets_b, lam = cutmix(inputs, targets, lam)

    outputs = net(inputs)
    loss_orig = criterion(outputs, targets)
    loss_mixed = criterion(outputs, targets_b)

    loss = lam * loss_orig + (1 - lam) * loss_mixed
    return loss, outputs


class NoMixUp(object):
    def __init__(self, criterion):
        self.criterion = criterion

    def step(self, model, inputs, targets):
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs


class MixUp(object):
    def __init__(self, criterion, alpha=0.2, max_lambda=True):
        self.alpha = alpha
        self.max_lamda = max_lambda
        self.criterion = criterion

    def step(self, model, inputs, targets):
        return mixup_pass_v2(model, self.criterion, inputs, targets, alpha=self.alpha, max_lambda=self.max_lamda)


class CutMix(MixUp):
    def __init__(self, criterion, alpha=0.2, max_lambda=True):
        super().__init__(criterion, alpha=alpha, max_lambda=max_lambda)

    def step(self, model, inputs, targets):
        return cutmix_pass(model, self.criterion, inputs, targets, alpha=self.alpha, max_lambda=self.max_lamda)
