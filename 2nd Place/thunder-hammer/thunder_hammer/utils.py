import cv2
from tqdm import tqdm
import multiprocessing
import functools
import os
import os.path as osp
import pydoc
import random
import socket
import time
from getpass import getuser
from socket import gethostname

import numpy as np

try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

from typing import Dict
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yaml


def update_base(config_base, config_update):
    for k, v in config_update.items():
        if k == "stages":
            config_base[k] = v

        elif isinstance(v, Dict):
            for k1, v1 in v.items():
                print(k, k1, v)
                config_base[k][k1] = v1

        else:
            config_base.update({k: v})
    return config_base


def update_config(config, params):
    for k, v in params.items():
        *path, key = k.split(".")
        if "." in k:
            key1, key2 = k.split(".")  # TODO fixme in general keys
            config[key1][key2] = v
        else:
            config.update({k: v})
            print(f"Overwriting {k} = {v} (was {config.get(key)})")
    return config


def get_paths(correction="./"):
    pcname = socket.gethostname()

    base, diff, path = "../", "", None
    for n in range(4):
        yaml_path = f"{diff}configs/paths.yml"
        if osp.exists(yaml_path):
            path = osp.join(correction, yaml_path)
        diff += base

    with open(path, "r") as stream:
        data_config = yaml.safe_load(stream)

    path_config = data_config[pcname]
    # return {"data_confg":path_config}
    return path_config


def fit(**kwargs):
    try:
        with open("configs/base.yml") as cfg:
            base_config = yaml.load(cfg, Loader=yaml.FullLoader)
    except:
        with open("../configs/base.yml") as cfg:
            base_config = yaml.load(cfg, Loader=yaml.FullLoader)

    if "config" in kwargs.keys():
        cfg_path = kwargs["config"]
        with open(cfg_path) as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

        merged_cfg = update_base(base_config, cfg_yaml)
    else:
        merged_cfg = base_config

    # print("!", kwargs)
    update_cfg = update_config(merged_cfg, kwargs)
    path_cfg = get_paths()
    if path_cfg:
        update_cfg = update_config(update_cfg, path_cfg)
    return update_cfg


def set_determenistic(seed=666, precision=10):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def get_host_info():
    return "{}@{}".format(getuser(), gethostname())


def init_dist(backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    torch.distributed.init_process_group(backend=backend, **kwargs)


def get_dist_info():
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt

def sum_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


def parallel_map(
    func,
    iterable,
    num_workers: int = multiprocessing.cpu_count(),
    desc: str = None,
    unordered: bool = False,
    total: int = None,
):
    with multiprocessing.Pool(processes=num_workers) as pool:
        if unordered:
            generator = pool.imap_unordered(func, iterable)
        else:
            generator = pool.imap(func, iterable)
        if desc is not None:
            if total is None:
                total = len(iterable)
            generator = tqdm(generator, total=total, desc=desc)

        results = list(generator)
    return results


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise ValueError(f"Image not found at `{path}`")
    image = cv2.imread(path)
    assert image is not None, path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == "__main__":
    get_paths()
