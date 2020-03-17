from typing import Dict, Any
import torch
import math
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from thunder_hammer.utils import object_from_dict

func_zoo = {
    "cosine_decay": lambda epoch, step, len_epoch, total_epoch: 0.5
    * (math.cos(step * math.pi / (total_epoch * len_epoch)) + 1)
}


class CosineWarmRestart(object):
    def __init__(
        self, optimizer, func="cosine_decay", warmup=True, warmup_epoch=1, period=10, min_lr=1e-5, low_epoch=1
    ):
        self.base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))[0]
        self.optimizer = optimizer
        self.warmup = warmup
        self.warmup_epoch = warmup_epoch
        self.period = period
        self.cos_period = period - low_epoch
        self.low_epoch = low_epoch
        self.lr_func = func_zoo[func]
        self.min_lr = min_lr

    def cosine_step(self, current_epoch, global_step, len_epoch):
        if self.warmup and current_epoch < self.warmup_epoch:
            lr = self.base_lrs * float(1 + global_step) / (self.warmup_epoch * len_epoch)
        else:
            lr = self.base_lrs * self.lr_func(current_epoch, global_step, len_epoch, self.cos_period)

        lr = max(self.min_lr, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def step(self, current_epoch, global_step, len_epoch):
        current_epoch = current_epoch % self.period
        if current_epoch >= self.period - self.low_epoch:
            global_step = len_epoch * self.cos_period
        else:
            global_step = global_step % (self.period * len_epoch)
        return self.cosine_step(current_epoch, global_step, len_epoch)


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value



class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 initialize=True) -> None:
        super().__init__(optimizer, param_group_field="lr", initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs


def main():
    cfg_model = {"type": "torchvision.models.resnet.resnet34", "pretrained": False}
    cfg_optimizer = {"type": "torch.optim.SGD", "lr": 0.6, "weight_decay": 0.0001, "momentum": 0.9}
    model = object_from_dict(cfg_model)
    # print(model)
    optimizer = object_from_dict(cfg_optimizer, params=filter(lambda x: x.requires_grad, model.parameters()))

    len_epoch = 10
    total_epoch = 200
    warmup_epoch = 5

    scheduler = CosineWarmRestart(optimizer, warmup_epoch=warmup_epoch, period=total_epoch)

    x, y = [], []
    for i in range(0, total_epoch * len_epoch * 1, 10):
        epoch = int(i / len_epoch)
        # print(epoch)
        x.append(i / len_epoch)
        y.append(scheduler.step(epoch, i, len_epoch))

    plt.plot(x, y)
    plt.xlabel("step")
    plt.ylabel("LR")
    # plt.yscale('log')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
