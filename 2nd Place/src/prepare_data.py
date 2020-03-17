import numpy as np

from thunder_hammer.utils import fit, object_from_dict, set_determenistic, update_config
import matplotlib.pyplot as plt
from addict import Dict
from fire import Fire
from thunder_hammer.utils import get_paths
import pandas as pd
import os.path as osp
from glob import glob
from tqdm import tqdm

labels = [
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


def prepare_csv():
    paths = get_paths()
    dataset_path = paths["train_data.path"]
    print(paths)

    df_map = pd.read_csv(osp.join(dataset_path, "annotation/train_metadata.csv"))
    df_label = pd.read_csv(osp.join(dataset_path, "annotation/train_labels.csv"))
    labels = list(df_label.columns)[1:]
    print(labels)

    pairs = []
    for label in tqdm(labels, total=len(labels)):
        tdf_label = df_label[df_label[label] == 1]
        tdf_map = df_map[df_map["seq_id"].isin(tdf_label["seq_id"])]

        for stem in tdf_map["file_name"]:
            season = stem.split("/")[0]
            for part in range(1, 6):
                stem_fn = stem.replace(f"{season}/", f"512_{season}_{part}/")

                in_file = osp.join(dataset_path, stem_fn)
                if osp.exists(in_file):
                    pairs.append((in_file, label))

    paths, labels = zip(*pairs)
    df = pd.DataFrame()
    df["path"], df["label"] = paths, labels
    df.to_csv("../tables/data_tmp.csv", index=False)


def prepare_filelist():
    df = pd.read_csv("../tables/data_tmp.csv")
    print(df.shape)

    # for label in labels:
    #     tdf = df[df["label"] == label]
    #     print(tdf.shape)

    label2index = dict([(label, n) for n, label in enumerate(labels)])
    df["label"] = [label2index[label] for label in df["label"]]

    print(df.label.unique())

    valid = df[df["path"].str.contains("S10")]
    print(valid.shape)
    train = df[~df["path"].isin(valid["path"])]
    print(train.shape)

    train = train.sample(frac=1).reset_index(drop=True)
    train.to_csv("../tables/train_list.txt", sep=" ", index=False, header=False)
    valid.to_csv("../tables/val_list.txt", sep=" ", index=False, header=False)


def main():
    cfg = Dict(Fire(fit))
    set_determenistic(cfg.seed)

    add_dict = {"data": {"batch_size": 24}}
    add_dict = Dict(add_dict)

    print(add_dict, "\t")

    cfg = Dict(update_config(cfg, add_dict))

    print("\t")

    print(cfg.data)
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


if __name__ == "__main__":
    prepare_csv()
    prepare_filelist()
    # main()
    # print(len(labels))
