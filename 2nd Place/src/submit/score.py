import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

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
    subm = pd.read_csv("submission.csv")
    print(subm.head())

    df_true = pd.read_csv("/media/n01z3/ssd1_intel/dataset/wild/annotation/train_labels.csv")
    df_true = df_true[df_true["seq_id"].isin(subm["seq_id"])]
    print(df_true.head())

    y_pred = subm[labels].values
    y_true = df_true[labels].values

    print(y_pred.shape, y_true.shape)
    print(np.argmax(y_pred, axis=1)[:-15])
    print(np.argmax(y_true, axis=1)[:-15])
    print(log_loss(y_true, y_pred))
    print(accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)))


if __name__ == "__main__":
    main()
