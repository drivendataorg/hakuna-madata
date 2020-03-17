import os.path as osp

import numpy as np
import pandas as pd
from thunder_hammer.utils import get_paths
from tqdm import tqdm

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


def check_paths():
    paths = get_paths()
    dataset_path = paths["train_data.path"]

    df_map = pd.read_csv(osp.join(dataset_path, "annotation/train_metadata.csv"))
    print(df_map.head())

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
    print(df.head())
    print(df.shape)
    df.to_csv(osp.join(dataset_path, "annotation/all_paths.csv"), index=False)


def prepare_filelist():
    paths = get_paths()
    dataset_path = paths["train_data.path"]

    df_meta = pd.read_csv(osp.join(dataset_path, "annotation/all_paths.csv"))
    df_labels = pd.read_csv(osp.join(dataset_path, "annotation/train_labels.csv"))
    df_labels = df_labels[df_labels["seq_id"].isin(df_meta["seq_id"])]

    seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])
    labels_arr = df_labels[LABELS].values

    index_arr = np.nonzero(labels_arr)
    print(index_arr[:10])
    print([seq2index.get(seq) for seq in df_meta["seq_id"][:10]])
    tmp = [np.nonzero(labels_arr[seq2index.get(seq), :]) for seq in df_meta["seq_id"]]  # df_meta['labels']

    df_meta["labels"] = [" ".join(list(el[0].astype(str))) for el in tmp]
    print(df_meta.head())
    df_meta.to_csv(osp.join(dataset_path, "annotation/file_list.csv"), index=False)


def check_filelist():
    paths = get_paths()
    dataset_path = paths["train_data.path"]

    df_meta = pd.read_csv(osp.join(dataset_path, "annotation/file_list.csv"))
    df_meta = df_meta[df_meta["labels"].str.contains(" ")]
    print(df_meta.head())
    print(df_meta.shape)


def filter_filelist():
    paths = get_paths()
    dataset_path = paths["train_data.path"]

    df_meta = pd.read_csv(osp.join(dataset_path, "annotation/file_list.csv"))
    print(df_meta.head())

    df_loss = pd.read_csv("tmp/train_loss.csv")
    print(df_loss.shape)
    print(df_loss.head())

    thr_loss = df_loss.loc[int(0.8 * df_loss.shape[0]), "loss"]  # thr_loss = 0.0012901991140097382 #
    print(thr_loss)

    high = df_loss[df_loss["loss"] > thr_loss]
    low = df_loss[df_loss["loss"] < thr_loss]

    low = low.sample(frac=0.1)

    df_loss = pd.concat([low, high])
    print(df_loss.shape)

    df_filter = df_meta[df_meta["file_name"].isin(df_loss["ids"])]
    print(df_filter.shape)
    df_filter.to_csv(osp.join(dataset_path, "annotation/file_list_filter.csv"), index=False)


if __name__ == "__main__":
    # main()
    # check_paths()
    # combine_labels()
    # prepare_filelist()
    # check_filelist()
    filter_filelist()
