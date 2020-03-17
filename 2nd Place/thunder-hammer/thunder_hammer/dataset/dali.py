import math
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from addict import Dict
from fire import Fire
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from tqdm import tqdm

from thunder_hammer.utils import fit, object_from_dict, set_determenistic, update_config

DATA_BACKEND_CHOICES = ['pytorch', 'syntetic']
DATA_BACKEND_CHOICES.append('dali-gpu')
DATA_BACKEND_CHOICES.append('dali-cpu')


class HybridPipe(dali.pipeline.Pipeline):
    def __init__(
        self,
        mode,
        path,
        batch_size=32,
        workers=4,
        crop_size=224,
        color_twist=True,
        local_rank=0,
        word_size=1,
        min_area=0.08,
        filelist=False,
        dali_cpu=False
    ):
        super(HybridPipe, self).__init__(
            batch_size=batch_size, num_threads=workers, device_id=local_rank, seed=12 + local_rank
        )

        self.dali_cpu = False
        self.mode = mode
        self.local_rank = local_rank
        self.world_size = word_size

        if filelist:
            filelist = osp.join(filelist, f"{mode}_list.txt")
            file_root = "/"
        else:
            filelist = ""
            file_root = osp.join(path, mode)

        print(filelist)

        self.input = dali.ops.FileReader(
            file_root=file_root,
            file_list=filelist,
            random_shuffle=mode == "train",
            shard_id=self.local_rank,
            num_shards=self.world_size,
            read_ahead=True,
        )

        self.color_twist = False
        self.jitter = False

        if self.mode == "train":
            # self.decode = dali.ops.ImageDecoderRandomCrop(
            #     device="cpu" if self.dali_cpu else "mixed",
            #     output_type=dali.types.RGB,
            #     random_aspect_ratio=[0.75, 1.25],
            #     random_area=[min_area, 1.0],
            #     num_attempts=100,
            # )

            if dali_cpu:
                dali_device = "cpu"
                self.decode = dali.ops.ImageDecoder(device=dali_device, output_type=dali.types.RGB)
            else:
                dali_device = "gpu"
                self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB, device_memory_padding=211025920,
                                               host_memory_padding=140544512)

            self.resize = dali.ops.RandomResizedCrop(
                device=dali_device,
                size=[crop_size, crop_size],
                interp_type=dali.types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 4. / 3.],
                random_area=[min_area, 1.0],
                num_attempts=100)

            # resize doesn't preserve aspect ratio on purpose
            # works much better with INTERP_TRIANGULAR
            # self.resize = dali.ops.Resize(
            #     device="cpu" if self.dali_cpu else "gpu",
            #     interp_type=dali.types.INTERP_TRIANGULAR,
            #     resize_x=crop_size,
            #     resize_y=crop_size,
            # )

            if color_twist:
                self.color_twist = dali.ops.ColorTwist(device="gpu")
                self.jitter = dali.ops.Jitter(device="gpu")

        else:
            self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB)
            # 14% bigger and dividable by 16 then center crop
            val_size = math.ceil((crop_size * 1.14 + 8) // 16 * 16)
            self.resize = dali.ops.Resize(
                device="gpu", interp_type=dali.types.INTERP_TRIANGULAR, resize_shorter=val_size
            )

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(crop_size, crop_size),
            image_type=dali.types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout=dali.types.NCHW,
        )
        self.random_flip = dali.ops.CoinFlip()
        self.range1 = dali.ops.Uniform(range=[0, 1])
        self.range2 = dali.ops.Uniform(range=[0.85, 1.15])
        self.range3 = dali.ops.Uniform(range=[-15, 15])

    def define_graph(self):
        # Read images and labels
        jpegs, labels = self.input(name="Reader")

        # Decode and augmentation
        images = self.decode(jpegs)
        images = self.resize(images)

        if self.dali_cpu:
            images = images.gpu()
        if self.mode == "train":
            if self.color_twist:
                images = self.color_twist(
                    images,
                    saturation=self.range2(),
                    contrast=self.range2(),
                    brightness=self.range2(),
                    hue=self.range3(),
                )
            images = self.normalize(
                images, mirror=self.random_flip(), crop_pos_x=self.range1(), crop_pos_y=self.range1()
            )
        else:
            images = self.normalize(images)

        return images, labels.gpu()


class DaliLoader(object):
    def __init__(
        self,
        mode,
        path,
        batch_size,
        workers,
        crop_size,
        color_twist=True,
        val_batch=False,
        min_area=0.08,
        filelist=False,
        dali_cpu=False,
    ):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.batch_size = batch_size
        if mode != "train" and val_batch:
            self.batch_size = val_batch

        self.mode = mode
        self.local_rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self.pipe = HybridPipe(
            mode=self.mode,
            path=path,
            batch_size=self.batch_size,
            workers=workers,
            local_rank=self.local_rank,
            word_size=self.world_size,
            crop_size=crop_size,
            color_twist=color_twist,
            min_area=min_area,
            filelist=filelist,
            dali_cpu=dali_cpu
        )

        self.pipe.build()

        self.loader = DALIClassificationIterator(
            self.pipe,
            size=self.pipe.epoch_size("Reader") / self.world_size,
            auto_reset=True,
            fill_last_batch=False,  # want real accuracy on validiation
            last_batch_padded=True,  # want epochs to have the same length
        )

    def __len__(self):
        return self.loader._size // self.loader.batch_size  # -1 for pytorch_lightning

    def __iter__(self):
        return ((batch[0]["data"], batch[0]["label"].squeeze().long()) for batch in self.loader)

    def sampler(self):
        return torch.utils.data.distributed.DistributedSampler  # for pytorch_lightning

    def dataset(self):  # for pytorch_lightning
        return None


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    set_determenistic(cfg.seed)

    add_dict = {"data": {"batch_size": 25}}
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
