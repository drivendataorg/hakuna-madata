from tqdm import tqdm
import os
import os.path as osp
from thunder_hammer.utils import get_paths
import pandas as pd


def main():
    paths = get_paths()
    dataset_path = paths["data.path"]

    val_season = "S10"

    # df_map = pd.read_csv(osp.join(dataset_path, "annotation/train_metadata.csv"))
    # df_label = pd.read_csv(osp.join(dataset_path, "annotation/train_labels.csv"))

    # df_map = df_map[df_map["file_name"].str.contains(val_season)]
    # os.makedirs('tmp', exist_ok=True)
    # df_map.to_csv('tmp/val.csv', index=False)
    df_map = pd.read_csv("tmp/val.csv")
    print(df_map.head())

    out = []
    for file_name, seq_id in tqdm(zip(df_map["file_name"], df_map["seq_id"]), total=df_map.shape[0]):
        for i in range(1, 5):
            full_path = file_name.replace(f"{val_season}/", f"512_{val_season}_{i}/")
            full_path = osp.join(dataset_path, full_path)
            if osp.exists(full_path):
                out.append((full_path.replace(f"{dataset_path}/", "../"), seq_id))
                # out_seq.append(seq_id)
                # print("ololo")
                break

    filenames, seqs = zip(*out)
    df = pd.DataFrame()
    df["file_name"] = filenames
    df["seq_id"] = seqs
    print(df.head())
    print(df.shape)
    df = df.head(100000)
    df.to_csv(osp.join(dataset_path, "annotation/valid_metadata_100k.csv"), index=False)


if __name__ == "__main__":
    main()
