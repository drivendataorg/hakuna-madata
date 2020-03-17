from tqdm import tqdm
import math
import os.path as osp
from addict import Dict
from fire import Fire

from thunder_hammer.utils import fit, object_from_dict, set_determenistic, update_config

import matplotlib.pyplot as plt

from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import torch

# import types
import numpy as np
import nvidia.dali.ops as ops
import pandas as pd
from nvidia import dali

# batch_size = 16
from thunder_hammer.utils import get_paths

PATHS = get_paths()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class ExternalInputIterator(object):
    def __init__(self, mode, data_path, batch_size):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"
        self.path = data_path
        self.batch_size = batch_size

        if mode == "train":
            ann_filename = "annotation/file_list_filter.csv"
        else:
            ann_filename = "annotation/file_list.csv"

        df_meta = pd.read_csv(osp.join(self.path, ann_filename))
        df_meta = df_meta.sample(frac=1).reset_index(drop=True)

        valid = df_meta[df_meta["file_name"].str.contains("_S10_")]
        if mode == "train":
            df_meta = df_meta[~df_meta["file_name"].isin(valid["file_name"])]
        elif mode == "val":
            df_meta = valid

        if mode == "val":
            df_meta = df_meta.sort_values("file_name").groupby("seq_id").first().reset_index()

        df_meta.reset_index(inplace=True)
        self.df_meta = df_meta

    def __iter__(self):
        self.i = 0
        self.n = self.df_meta.shape[0]
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            row = self.df_meta.iloc[self.i]
            jpeg_filename = osp.join(self.path, row["file_name"])
            targets = np.zeros(54, dtype=np.uint8)
            for index in row["labels"].split(" "):
                targets[int(index)] = 1

            f = open(jpeg_filename, "rb")
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(targets)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    def __len__(self):
        return self.df_meta.shape[0]

    next = __next__


class ExternalSourcePipeline(dali.pipeline.Pipeline):
    def __init__(
        self,
        mode,
        path,
        batch_size=32,
        workers=4,
        crop_size=224,
        long_side=512,
        color_twist=True,
        local_rank=0,
        word_size=1,
        min_area=0.2,
    ):

        super(ExternalSourcePipeline, self).__init__(batch_size, workers, local_rank, seed=12 + local_rank)
        self.input_image = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        self.color_twist = False
        self.jitter = False

        self.mode = mode

        self.eii = ExternalInputIterator(mode=self.mode, data_path=path, batch_size=batch_size)
        self.iterator = iter(self.eii)

        if self.mode == "train":
            self.decode = dali.ops.ImageDecoderRandomCrop(
                device="mixed",
                output_type=dali.types.RGB,
                random_aspect_ratio=[0.75, 1.25],
                random_area=[min_area, 1.0],
                num_attempts=100,
            )
            self.resize = dali.ops.Resize(
                device="gpu", interp_type=dali.types.INTERP_LANCZOS3, resize_x=crop_size, resize_y=crop_size,
            )

            if color_twist:
                self.color_twist = dali.ops.ColorTwist(device="gpu")

            self.normalize = dali.ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=dali.types.FLOAT,
                crop=(crop_size, crop_size),
                image_type=dali.types.RGB,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                output_layout=dali.types.NCHW,
            )

        else:
            self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB)
            self.resize = dali.ops.Resize(
                device="gpu", interp_type=dali.types.INTERP_LANCZOS3, resize_longer=long_side
            )

            self.normalize = dali.ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=dali.types.FLOAT,
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
        self.jpegs = self.input_image()
        self.labels = self.input_label()

        images = self.decode(self.jpegs)
        images = self.resize(images)

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

        return images, self.labels

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels)

    def __len__(self):
        return len(self.eii)


class DaliLoader(object):
    def __init__(
        self, mode, path, batch_size=16, workers=4, crop_size=224, long_side=512, color_twist=True, min_area=0.2,
    ):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.batch_size = batch_size

        self.mode = mode
        self.local_rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self.pipe = ExternalSourcePipeline(
            mode=self.mode,
            path=path,
            batch_size=self.batch_size,
            workers=workers,
            local_rank=self.local_rank,
            word_size=self.world_size,
            crop_size=crop_size,
            long_side=long_side,
            color_twist=color_twist,
            min_area=min_area,
        )

        self.pipe.build()

        self.loader = DALIClassificationIterator(
            self.pipe,
            size=len(self.pipe) / self.world_size,
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


def main1():
    pipe = ExternalSourcePipeline(mode="val", path=PATHS["data.path"], batch_size=8, workers=8, local_rank=0)
    pipe.build()
    pipe_out = pipe.run()

    batch_cpu = pipe_out[0].as_cpu()
    labels_cpu = pipe_out[1]

    img = batch_cpu.at(2)
    print(img.shape)
    print(labels_cpu.at(2))

    image = np.transpose(img, (1, 2, 0))
    image_show = np.uint8(255 * (IMAGENET_STD * image + IMAGENET_MEAN))

    plt.imshow(image_show)
    plt.show()


def main():
    cfg = Dict(Fire(fit))
    set_determenistic(cfg.seed)

    add_dict = {"train_data": {"batch_size": 8}}
    add_dict = Dict(add_dict)

    print(add_dict, "\t")

    cfg = Dict(update_config(cfg, add_dict))

    print("\t")

    print(cfg.data)
    loader = object_from_dict(cfg.train_data)

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
            plt.subplot(2, 4, i + 1)
            shw = np.uint8(np.clip(255 * (imagenet_mean * img[i] + imagenet_std), 0, 255))
            plt.imshow(shw)

        plt.show()

        # break


if __name__ == "__main__":
    main()
