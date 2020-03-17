# from tqdm import tqdm
import logging
import os.path as osp

import numpy as np
import pandas as pd
import torch
from PIL import Image

# from torch.utils.data import DataLoader

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)
from thunder_hammer.utils import get_paths

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


class HakunaInferDataset:
    def __init__(self, mode, data_path, long_side):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"
        self.path = data_path
        self.long_side = long_side
        self.mode = mode

        if self.mode == "test":
            df_meta = pd.read_csv(osp.join(osp.abspath(__file__)), "data/test_metadata.csv")
            self.path = "data"

        else:
            df_path = osp.join(self.path, "annotation/valid.csv")
            df_meta = pd.read_csv(df_path)

        self.groups = list(df_meta.groupby("seq_id"))

        if self.mode == "val":
            df_labels = pd.read_csv(osp.join(self.path, "annotation/train_labels.csv"))
            df_labels = df_labels[df_labels["seq_id"].isin(df_meta["seq_id"])]
            self.labels = df_labels[LABELS].values
            self.seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])

    def get_image(self, full_path):
        img = Image.open(full_path)
        w, h = img.size
        ratio = max(h / self.long_side, w / self.long_side)
        # want them to be divizable by 16
        new_w = int((w / ratio) // 16 * 16)
        new_h = int((h / ratio) // 16 * 16)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        np_img = np.asarray(img, dtype=np.uint8)
        np_img = np_img - np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
        np_img = np_img / np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
        np_img = np.rollaxis(np_img, 2)
        img_tensor = torch.from_numpy(np_img)
        return img_tensor

    def __getitem__(self, idx):
        batch = {}

        seq_id, group_df = self.groups[idx]
        batch["seq_id"] = seq_id
        images = []
        for file_name in group_df["file_name"]:
            images.append(self.get_image(osp.join(self.path, file_name)))
        batch["images"] = torch.stack(images)
        if self.mode == "val":
            batch["label"] = self.labels[self.seq2index.get(seq_id)]

        return batch

    def __len__(self):
        return len(self.groups)


def InferLoader(
    mode,
    path,
    batch_size=16,
    workers=8,
    crop_size=224,
    long_side=512,
    color_twist=True,
    val_batch=False,
    min_area=0.2,
):
    assert mode in ["train", "val", "test"], f"unknown mode {mode}"
    batch_size = 1
    mode = mode
    local_rank = 0
    world_size = 1
    sampler = None
    path = path
    long_side = long_side
    workers = workers

    dataset = HakunaInferDataset(mode=mode, data_path=path, long_side=long_side)

    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    return loader


def make_val():
    path = get_paths()["train_data.path"]
    df_path = osp.join(path, "annotation/all_paths.csv")
    df_meta = pd.read_csv(df_path)
    valid = df_meta[df_meta["file_name"].str.contains("_S10_")]

    df_path = osp.join(path, "annotation/valid.csv")

    valid.to_csv(df_path, index=False)

    df_meta = pd.read_csv(df_path)
    print(df_meta)

    test_metadata = df_meta.groupby("seq_id").first().reset_index()

    print(test_metadata.head())

    groups = df_meta.groupby("seq_id")
    print(len(groups))
    for sample_id, group in groups:
        print(sample_id)
        print(group)


def make_tst():
    path = get_paths()["train_data.path"]
    df_path = osp.join(path, "annotation/all_paths.csv")
    df_meta = pd.read_csv(df_path)
    valid = df_meta[df_meta["file_name"].str.contains("_S10_")]

    valid = valid.head(1000)

    valid.to_csv(osp.join(path, "test_metadata.csv"), index=False)
    valid = valid.drop_duplicates("seq_id")

    sample = pd.DataFrame()
    sample["seq_id"] = valid["seq_id"]
    for label in LABELS:
        sample[label] = [0] * sample.shape[0]

    sample.to_csv(osp.join(path, "submission_format.csv"), index=False)


def check_iter():
    paths = get_paths()
    vloader = InferLoader(mode="val", path=paths["train_data.path"], batch_size=1)
    print(len(vloader))
    for idx, batch in enumerate(vloader):
        print(batch["seq_id"])
        print(batch["images"][0].shape)
        print(batch["label"])

        break


if __name__ == "__main__":
    # check_iter()
    # make_val()
    make_tst()
