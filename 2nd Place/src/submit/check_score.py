import thunder_hammer
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
import os.path as osp

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


def main():
    criterion = thunder_hammer.smooth.MultiLabelSoftMarginLoss()

    subm = pd.read_csv("submission.csv")
    print(subm.head())

    df_meta = pd.read_csv("/media/n01z3/ssd1_intel/dataset/wild/test_metadata.csv")
    df_labels = pd.read_csv(osp.join("/media/n01z3/ssd1_intel/dataset/wild/", "annotation/train_labels.csv"))
    df_labels = df_labels[df_labels["seq_id"].isin(df_meta["seq_id"])]
    labels = df_labels[LABELS].values
    seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])

    y_preds = subm[LABELS].values

    losses = []
    for i, seq_id in enumerate(subm["seq_id"]):
        targets = torch.from_numpy(labels[seq2index.get(seq_id)]).unsqueeze(0)
        output = torch.from_numpy(y_preds[i]).unsqueeze(0)
        logits = torch.log(output / (1 - output + 1e-7)).type(torch.FloatTensor)
        # print(targets)
        # print(logits)

        loss = criterion(logits, targets)
        losses.append(loss.data.numpy())

    print(np.mean(losses))


if __name__ == "__main__":
    main()
