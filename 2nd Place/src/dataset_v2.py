from PIL import ImageFile
from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import logging
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

try:
    import albumentations as alb
    import jpeg4py as jpeg
    from thunder_hammer.utils import get_paths
except:
    logging.info("looks like we start test, lol")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
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
RARE = [
    "bat",
    "steenbok",
    "cattle",
    "zorilla",
    "duiker",
    "civet",
    "genet",
    "rhinoceros",
    "honeybadger",
    "wildcat",
    "rodents",
    "caracal",
    "vulture",
    "hyenastriped",
    "reptiles",
    "bushbuck",
    "aardwolf",
    "leopard",
    "porcupine",
    "aardvark",
]


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        img = np.clip(img, 0, 255)
        nump_array = np.uint8(img)

        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def prepare_paths(df_path):
    paths = get_paths()
    dataset_path = paths["data.path"]

    df_map = pd.read_csv(osp.join(dataset_path, "annotation/train_metadata.csv"))
    df_labels = pd.read_csv(osp.join(dataset_path, "annotation/train_labels.csv"))
    print(df_labels.head())

    out = []
    for file_name, seq_id in tqdm(zip(df_map["file_name"], df_map["seq_id"]), total=df_map.shape[0]):
        for i in range(1, 5):
            season = file_name.split("/")[0]
            full_path = file_name.replace(f"{season}/", f"512_{season}_{i}/")
            full_path = osp.join(dataset_path, full_path)
            if osp.exists(full_path):
                out.append((full_path.replace(f"{dataset_path}/", ""), seq_id))
                break

    filenames, seqs = zip(*out)
    df = pd.DataFrame()
    df["file_name"] = filenames
    df["seq_id"] = seqs
    df["label"] = [0] * df.shape[0]

    tdf = df_labels[df_labels["empty"] == 1]
    df.loc[df["seq_id"].isin(tdf["seq_id"]), "label"] = 1

    for label in RARE:
        tdf = df_labels[df_labels[label] == 1]
        df.loc[df["seq_id"].isin(tdf["seq_id"]), "label"] = 2

    print(df.head(100))
    print(df.shape)
    print(np.sum(df["label"].values))
    df.to_csv(df_path, index=False)


def train_aug(height=224, width=224):
    return alb.Compose(
        [
            alb.RandomResizedCrop(height=height, width=width, always_apply=True, interpolation=cv2.INTER_LANCZOS4),
            alb.CLAHE(p=0.1),
            alb.ToGray(p=0.2),
            alb.RandomBrightnessContrast(p=0.6),
        ]
    )


def val_aug(height=224, width=224):
    return alb.Compose([alb.Resize(height=height, width=width, always_apply=True, interpolation=cv2.INTER_LANCZOS4),])


class HakunaDatasetFast:
    def __init__(self, mode, data_path, long_side, crop_size=256):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"

        self.path = data_path
        self.long_side = long_side

        if mode == "test":
            df_meta = pd.read_csv(osp.join(osp.abspath(__file__)), "data/test_metadata.csv")
            self.path = "data"
        else:
            df_path = osp.join(self.path, "annotation/all_paths.csv")
            if not osp.exists(df_path):
                prepare_paths(df_path)

            df_meta = pd.read_csv(df_path)

            valid = df_meta[df_meta["file_name"].str.contains("_S10_")]
            if mode == "train":
                df_meta = df_meta[~df_meta["file_name"].isin(valid["file_name"])]
            elif mode == "val":
                df_meta = valid

            if mode == "val":
                df_meta = df_meta.sort_values("file_name").groupby("seq_id").first().reset_index()

            # else:
            # print(df_meta.shape, 'meta oringal')
            # other = df_meta[df_meta["label"] == 0]
            # empty = df_meta[df_meta["label"] == 1]
            # rares = df_meta[df_meta["label"] == 2]
            # other = other.sample(frac=0.5)
            # df_meta = pd.concat([other, empty] + 3 * [rares])

            # print(df_meta.shape)

            df_labels = pd.read_csv(osp.join(self.path, "annotation/train_labels.csv"))
            df_labels = df_labels[df_labels["seq_id"].isin(df_meta["seq_id"])]
            self.labels = df_labels[LABELS].values
            self.seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])

        self.df_meta = df_meta
        self.aug = False
        if mode == "train":
            self.aug = train_aug(int(0.75 * crop_size), crop_size)
        else:
            self.aug = val_aug(int(0.75 * crop_size), crop_size)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        img = Image.open(osp.join(self.path, row["file_name"]))
        w, h = img.size
        ratio = max(h / self.long_side, w / self.long_side)
        new_w = int((w / ratio) // 16 * 16)
        new_h = int((h / ratio) // 16 * 16)
        img = img.resize((new_w, new_h), Image.ANTIALIAS)

        # image = cv2.imread(osp.join(self.path, row["file_name"]))[:, :, ::-1]
        # ratio = max(image.shape[0] / self.long_side, image.shape[1] / self.long_side)
        # image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        seq_id = row["seq_id"]
        target = self.labels[self.seq2index.get(seq_id)]

        image = np.asarray(img, dtype=np.uint8)
        if self.aug:
            image = self.aug(image=image)["image"]
        return image, target

    def __len__(self):
        return self.df_meta.shape[0]


class HakunaPrefetchedLoader(object):
    def __init__(
        self,
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

        self.batch_size = batch_size
        self.mode = mode
        self.local_rank = 0
        self.world_size = 1
        self.sampler = None
        self.shuffle = mode == "train"

        self.mode = mode
        self.path = path
        self.long_side = long_side
        self.crop_size = crop_size
        self.workers = workers

        self.create_loader()

    def create_loader(self):
        dataset = HakunaDatasetFast(
            mode=self.mode, data_path=self.path, long_side=self.long_side, crop_size=self.crop_size
        )

        if torch.distributed.is_initialized():
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            self.shuffle = False

        self.loader = torch.utils.data.DataLoader(
            dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.workers,
            collate_fn=fast_collate,
            # pin_memory=True
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

    def reset(self):
        self.create_loader()


def main():
    path = get_paths()
    val_dataloader = HakunaPrefetchedLoader(mode="val", path=path["train_data.path"], long_side=256, batch_size=8)
    print(len(val_dataloader))

    for idx, batch in enumerate(val_dataloader):
        images, targets = batch
        images = images.cpu().numpy()
        targets = targets.cpu().numpy()
        plt.figure()
        for i in range(images.shape[0]):
            plt.subplot(2, 4, i + 1)
            image = np.transpose(images[i], (1, 2, 0))
            image_show = np.uint8(255 * (IMAGENET_STD * image + IMAGENET_MEAN))
            plt.title(np.argmax(targets[i]))
            plt.imshow(image_show)

        plt.show()


if __name__ == "__main__":
    main()
    # path = get_paths()
    # prepare_paths(df_path=path["data.path"] + "/tmp.csv")
