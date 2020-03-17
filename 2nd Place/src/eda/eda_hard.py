import os.path as osp

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from thunder_hammer.utils import get_paths

path = get_paths()
data_path = get_paths()["train_data.path"]

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


def main():
    lst = []
    for i in range(3):
        df = pd.read_csv(f"../tmp/{i}_loss.csv")
        print(df.shape)
        lst.append(df)

    df = pd.concat(lst)
    print(df.head())

    df.sort_values("loss", inplace=True)
    print(df.head())
    print(df.tail())
    df.to_csv("../tmp/train_loss.csv", index=False)


def eda_hard():
    df_meta = pd.read_csv(osp.join(data_path, "annotation/file_list.csv"))
    print(df_meta.head())

    path2label = dict(zip(df_meta["file_name"], df_meta["labels"]))

    df = pd.read_csv("tmp/val_loss.csv")
    for i in range(6 * 12):
        file_name = df.loc[df.shape[0] - i - 1, "ids"]
        loss = df.loc[df.shape[0] - i - 1, "loss"]
        file_path = osp.join(data_path, file_name)

        label_str = path2label.get(file_name)
        label_names = " ".join([labels[int(index)] for index in label_str.split(" ")])

        print(file_path)

        image = cv2.imread(file_path)[:, :, ::-1]

        plt.subplot(6, 12, i + 1)
        plt.title(f"{label_names} | {loss:0.3f}")
        plt.imshow(image)

    plt.show()


def plot_hard():
    df = pd.read_csv("../tmp/train_loss.csv")
    plt.plot(df["loss"].values)
    plt.ylabel("loss")
    plt.xlabel("sample_id")
    plt.show()


if __name__ == "__main__":
    # main()
    # eda_hard()
    plot_hard()
