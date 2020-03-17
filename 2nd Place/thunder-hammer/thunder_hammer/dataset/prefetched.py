import math
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from addict import Dict
from fire import Fire
from tqdm import tqdm



from thunder_hammer.utils import fit, object_from_dict, set_determenistic, update_config


class PrefetchedLoader(object):
    def __init__(self, mode, path, batch_size, workers, crop_size, color_twist=True, val_batch=False):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.batch_size = batch_size
        if mode != "train" and val_batch:
            self.batch_size = val_batch

        self.mode = mode
        self.local_rank = 0
        self.world_size = 1
        self.sampler = None
        self.shuffle = mode == "train"
        if mode == "train":
            self.transform_lst = [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip()]
            if color_twist:
                self.transform_lst.append(
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08)
                )
        else:
            val_size = math.ceil((crop_size * 1.14 + 8) // 16 * 16)
            self.transform_lst = [transforms.Resize(val_size), transforms.CenterCrop(crop_size)]

        dataset = datasets.ImageFolder(osp.join(path, mode), transforms.Compose(self.transform_lst))

        if torch.distributed.is_initialized():
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            self.shuffle = False

        self.loader = torch.utils.data.DataLoader(
            dataset,
            sampler=self.sampler,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=workers,
            collate_fn=self.fast_collate,
        )

    def prefetch(self):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)

                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self.prefetch()

    def dataset(self):  # for pytorch_lightning
        return None

    @staticmethod
    def fast_collate(batch):
        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        w = imgs[0].size[0]
        h = imgs[0].size[1]
        tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            if nump_array.ndim < 3:
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)

            tensor[i] += torch.from_numpy(nump_array)

        return tensor, targets


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    set_determenistic(cfg.seed)

    add_dict = {"data": {"batch_size": 25, "type": "thunder_hammer.dataset.prefetched.PrefetchedLoader"}}
    add_dict = Dict(add_dict)

    print(add_dict, "\t")

    cfg = Dict(update_config(cfg, add_dict))

    print("\t")

    print(cfg)
    loader = object_from_dict(cfg.data, mode="val")

    batch_size = loader.batch_size
    side = int(np.sqrt(batch_size))
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    for images, targets in tqdm(loader, total=len(loader)):
        print(images.shape)
        print(targets.shape)

        img = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        labels = targets.cpu().numpy()

        plt.figure(figsize=(25, 35))
        for i in range(batch_size):
            plt.subplot(side, side, i + 1)
            shw = np.uint8(np.clip(255 * (imagenet_mean * img[i] + imagenet_std), 0, 255))
            plt.imshow(shw)

        plt.show()

        break
