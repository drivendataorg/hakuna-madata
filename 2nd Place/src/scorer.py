import pandas as pd
import yaml
from scipy.stats import gmean
import numpy as np
import torch.nn as nn
import ttach as tta
import os
import warnings
from time import time
from functools import partial
import ttach.functional as F
from ttach.base import ImageOnlyTransform, Compose
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from addict import Dict
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from fire import Fire
from tqdm import tqdm

# from torch2trt import torch2trt
from thunder_hammer.utils import (
    fit,
    set_determenistic,
    object_from_dict,
    reduce_tensor,
    AverageMeter,
    to_python_float,
)


warnings.filterwarnings("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class ThreeCrops(ImageOnlyTransform):
    """Makes 4 crops for each corner + center crop

    Args:
        crop_height (int): crop height in pixels
        crop_width (int): crop width in pixels
    """

    def __init__(self, crop_height, crop_width):
        crop_functions = (
            partial(F.crop_lt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rb, crop_h=crop_height, crop_w=crop_width),
            partial(F.center_crop, crop_h=crop_height, crop_w=crop_width),
        )
        super().__init__("crop_fn", crop_functions)

    def apply_aug_image(self, image, crop_fn=None, **kwargs):
        return crop_fn(image)

    def apply_deaug_mask(self, mask, **kwargs):
        raise ValueError("`FiveCrop` augmentation is not suitable for mask!")


def six_crop_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), ThreeCrops(crop_height, crop_width)])


#
# def one_crop_transform(crop_height, crop_width):
#     return Compose([OneCrops(crop_height, crop_width)])


def get_models(hparams, distributed=False):
    models = []
    for weight in hparams.weights:
        model = object_from_dict(hparams.model)
        model = model.cuda()

        print(weight)
        checkpoint = torch.load(weight, map_location="cpu")
        if "state_dict" in checkpoint.keys():
            state_dict = checkpoint["state_dict"]
            sanitized = {}
            for k, v in state_dict.items():
                if "101" in weight:
                    sanitized[k.replace("model.model", "model")] = v
                else:
                    # sanitized[k.replace("model.model", "model")] =
                    sanitized[k.replace("model.", "")] = v

            model.load_state_dict(sanitized)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        if distributed:
            model = DDP(model, delay_allreduce=True)

        # model = tta.ClassificationTTAWrapper(model, tta.aliases.ten_crop_transform(224, 320), merge_mode='mean')
        # model = tta.ClassificationTTAWrapper(model, tta.aliases.hflip_transform(), merge_mode="mean")
        models.append(model)

    return models


def main(hparams, model_configs):
    if hparams.seed:
        set_determenistic(hparams.seed)

    distributed = False
    if "WORLD_SIZE" in os.environ:
        print("start distributed")
        distributed = int(os.environ["WORLD_SIZE"]) > 1
        local_rank = int(os.environ["RANK"])
        print(f"local_rank {local_rank}")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=local_rank)
        world_size = torch.distributed.get_world_size()

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu_id)
        world_size = 1
        local_rank = 1

    criterion = object_from_dict(hparams.loss)
    # criterion = nn.NLLLoss()
    models = []
    for model_hparams in model_configs:
        print(model_hparams)
        models += get_models(model_hparams, distributed=distributed)

    # for weight in hparams.weights:
    #     print(weight)
    #
    #     models.append(model)

    acc1 = [object_from_dict(el) for el in hparams.metrics][0]

    # lst = []
    with torch.no_grad():
        for el in hparams.data_test:
            hparams.val_data.type = el.type
            val_loader = object_from_dict(hparams.val_data, mode="val")
            name = el.type.split(".")[-1]

            losses = AverageMeter()
            top1 = AverageMeter()

            t0 = time()
            tloader = tqdm(val_loader, desc="acc1, loss", leave=True)
            for n, (inputs, targets) in enumerate(tloader):
                # print(ids)
                outputs = []
                for model in models:
                    output = model(inputs)
                    outputs.append(torch.sigmoid(output))

                arr = torch.stack(outputs, dim=0).cpu().numpy()
                out = gmean(arr, axis=0)
                output = torch.from_numpy(out).cuda()

                logits = torch.log(output / (1 - output + 1e-7))

                # for i in range(len(ids)):
                #     loss_i = criterion(logits[i:i+1], targets[i:i+1])
                #     loss_sample = to_python_float(loss_i.data)
                #     lst.append((ids[i], loss_sample))

                loss = criterion(logits, targets)

                if torch.isnan(loss.data):
                    print(
                        n,
                        "fuck",
                        np.amin(logits.cpu().numpy()),
                        np.amax(logits.cpu().numpy()),
                        np.amax(output.cpu().numpy()),
                    )
                    continue

                prec1 = acc1(output, targets)

                if distributed:
                    reduced_loss = reduce_tensor(loss.data, world_size)
                    prec1 = reduce_tensor(prec1, world_size)
                else:
                    reduced_loss = loss.data

                losses.update(to_python_float(reduced_loss), inputs.size(0))
                top1.update(to_python_float(prec1), inputs.size(0))

                tloader.set_description(f"acc1:{top1.avg:.1f} loss:{losses.avg:.4f}")
                tloader.refresh()

            if local_rank == 0:
                print(f"{name}\n acc1:{top1.avg:.3f}\t loss:{losses.avg:.5f}\n time:{time() - t0:.1f}\n")


if __name__ == "__main__":
    models = ["rx50_stages_7.yml"]  #'rx50_stages_2.yml', 'rx50_stages_2.yml', rx101_stages_2 "rx101_stages_7.yml",

    configs = []
    for model in models:
        with open(f"configs/{model}") as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)
        configs.append(Dict(cfg_yaml))

    cfg = Dict(Fire(fit))
    main(cfg, configs)
