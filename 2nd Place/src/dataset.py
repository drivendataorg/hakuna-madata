import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from thunder_hammer.utils import get_paths
from torch.utils.data import DataLoader

from addict import Dict
from fire import Fire

from thunder_hammer.utils import fit, object_from_dict, set_determenistic, update_config

PATHS = get_paths()
LABELS = [
    "aardvark",
    "aardwolf",
    "baboon",
    "bat",
    "batearedfox",
    "buffalo",
    "bushbuck",
    "caracal",
    "cattle",
    "cheetah",
    "civet",
    "dikdik",
    "duiker",
    "eland",
    "elephant",
    "empty",
    "gazellegrants",
    "gazellethomsons",
    "genet",
    "giraffe",
    "guineafowl",
    "hare",
    "hartebeest",
    "hippopotamus",
    "honeybadger",
    "hyenaspotted",
    "hyenastriped",
    "impala",
    "insectspider",
    "jackal",
    "koribustard",
    "leopard",
    "lionfemale",
    "lionmale",
    "mongoose",
    "monkeyvervet",
    "ostrich",
    "otherbird",
    "porcupine",
    "reedbuck",
    "reptiles",
    "rhinoceros",
    "rodents",
    "secretarybird",
    "serval",
    "steenbok",
    "topi",
    "vulture",
    "warthog",
    "waterbuck",
    "wildcat",
    "wildebeest",
    "zebra",
    "zorilla",
]


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    # ids = [target[2] for target in batch]
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


class HakunaDataset:
    def __init__(self, mode, path, long_side=512, crop_size=224, color_twist=True, min_area=0.2):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.path = path
        self.long_side = long_side
        df_paths = pd.read_csv(osp.join(self.path, "annotation/all_paths.csv"))
        valid = df_paths[df_paths["file_name"].str.contains("_S10_")].head(100000)
        # if mode == "train":
        #     df_paths = df_paths[~df_paths["file_name"].isin(valid["file_name"])]
        if mode == "val":
            df_paths = valid

        df_labels = pd.read_csv(osp.join(self.path, "annotation/train_labels.csv"))
        df_labels = df_labels[df_labels["seq_id"].isin(df_paths["seq_id"])]

        # if mode == "val":
        #     df_paths = df_paths.sort_values("file_name").groupby("seq_id").first().reset_index()
        #
        print(df_paths.shape)
        # print(df_paths.head())

        self.df_paths = df_paths
        self.labels = df_labels[LABELS].values
        self.seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])

        self.transform = False
        if isinstance(crop_size, str):
            crop_size = (int(crop_size.split(",")[0]), int(crop_size.split(",")[1]))

        if mode == "train":
            transform_lst = [
                transforms.RandomResizedCrop(crop_size, scale=(min_area, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
            if color_twist:
                transform_lst.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08))
            self.transform = transforms.Compose(transform_lst)

    def __getitem__(self, idx):
        row = self.df_paths.iloc[idx]
        img = Image.open((osp.join(self.path, row["file_name"])))
        # sample_id = row["file_name"]
        # resize to 512 longest size:
        w, h = img.size
        ratio = max(h / self.long_side, w / self.long_side)
        # want them to be divizable by 16
        new_w = int((w / ratio) // 16 * 16)
        new_h = int((h / ratio) // 16 * 16)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

        if self.transform:
            img = self.transform(img)

        seq_id = row["seq_id"]
        target = self.labels[self.seq2index.get(seq_id)]
        return img, target

    def __len__(self):
        return self.df_paths.shape[0]


class HakunaPrefetchedLoader(object):
    def __init__(
        self, mode, path, batch_size=16, workers=4, crop_size=224, long_side=512, color_twist=True, min_area=0.2,
    ):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.batch_size = batch_size
        self.mode = mode
        self.local_rank = 0
        self.world_size = 1
        self.sampler = None
        self.shuffle = mode == "train"

        dataset = HakunaDataset(
            mode=mode, path=path, crop_size=crop_size, color_twist=color_twist, min_area=min_area, long_side=long_side
        )

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
            collate_fn=fast_collate,
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


def main():
    # val_dataset = HakunaDataset(mode="val", path=PATHS["data.path"], long_side=320, crop_size=(192, 256))
    # val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=8, collate_fn=fast_collate)

    cfg = Dict(Fire(fit))
    set_determenistic(cfg.seed)

    add_dict = {"val_data": {"batch_size": 8}}
    add_dict = Dict(add_dict)

    print(add_dict, "\t")

    cfg = Dict(update_config(cfg, add_dict))

    print("\t")

    print(cfg.data)
    loader = object_from_dict(cfg.val_data)
    batch_size = loader.batch_size
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # for idx, batch in enumerate(loader):
    #     images, targets = batch
    #     images = images.numpy()
    #     targets = targets.numpy()
    #     plt.figure()
    #     for i in range(images.shape[0]):
    #         plt.subplot(2, 4, i + 1)
    #         image = np.transpose(images[i], (1, 2, 0))
    #         plt.title(np.argmax(targets[i]))
    #         plt.imshow(image)

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


if __name__ == "__main__":
    main()
