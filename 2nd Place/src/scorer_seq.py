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
from torchvision.models.resnet import resnext50_32x4d
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
from thunder_hammer.utils import fit, set_determenistic, object_from_dict, reduce_tensor, AverageMeter, to_python_float


warnings.filterwarnings("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def get_models(hparams, distributed=False):
    models = []
    for weight in hparams.weights:
        model = object_from_dict(hparams.model)
        # model = resnext50_32x4d(num_classes=54)
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

                sanitized[k.replace("model.last_linear.", "fc.")] = v

            model.load_state_dict(sanitized, strict=False)
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
    models = []
    path = "src/submit/assets/rx50_v7_s4_e6.pth"
    for model_hparams in model_configs:
        print(model_hparams)
        models += get_models(model_hparams, distributed=distributed)
        # models.append(torch.jit.load(str(path)).cuda())

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
            for n, batch in enumerate(tloader):
                targets = batch["label"].cuda()
                outputs = []
                for model in models:
                    imgs = batch["images"][0].type(torch.FloatTensor).cuda()
                    # imgs = batch["images"][0]  # .type(torch.FloatTensor).cuda()
                    # mirror = torch.flip(imgs, (3,))
                    # imgs_mirror = torch.cat([imgs, mirror], dim=0).type(torch.FloatTensor).cuda()
                    # print(imgs.shape)
                    output = torch.sigmoid(model(imgs))
                    arr = output.cpu().numpy()
                    model_predict = gmean(arr, axis=0)
                    outputs.append(model_predict)

                # arr = torch.stack(outputs, dim=0).cpu().numpy()
                arr = np.array(outputs)
                out = gmean(arr, axis=0)
                output = torch.from_numpy(out).cuda().unsqueeze(0)

                logits = torch.log(output / (1 - output + 1e-7))
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

                losses.update(to_python_float(reduced_loss), 1)
                top1.update(to_python_float(prec1), 1)

                tloader.set_description(f"acc1:{top1.avg:.1f} loss:{losses.avg:.4f}")
                tloader.refresh()

            if local_rank == 0:
                print(f"{name}\n acc1:{top1.avg:.3f}\t loss:{losses.avg:.5f}\n time:{time() - t0:.1f}\n")


if __name__ == "__main__":
    models = [
        "rx50_stages_7.yml",
        "rx101_stages_7.yml",
    ]  #'rx50_stages_2.yml', 'rx50_stages_2.yml', rx101_stages_2 "rx101_stages_7.yml",

    configs = []
    for model in models:
        with open(f"configs/{model}") as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)
        configs.append(Dict(cfg_yaml))

    cfg = Dict(Fire(fit))
    main(cfg, configs)
