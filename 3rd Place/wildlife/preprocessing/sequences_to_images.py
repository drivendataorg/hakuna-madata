# extract background and mean of sequiences
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import os
import multiprocessing
import random


class PreProcessor:

    def __init__(self, dataset_path):
        self.processors = 3 # set num cores-1
        # train_metadata and  train_labels path
        self.DATA_PATH = Path(dataset_path)
        # background path ~130Gb
        self.BACKGROUND_DATA_PATH = self.DATA_PATH / 'preprocessed' / 'background'
        # mean path ~400Gb
        self.MEAN_DATA_PATH = self.DATA_PATH / 'preprocessed' / 'mean'

    def process(self, group):
        try:
            if random.randint(0, 100000) == 1:
                print('processing', group[0])
            is_empty = "empty" if group[1][0] == 1 else "animal"
            season = group[3][0]
            trainval = "val" if season[-1] in ["9", "0"] else "train"
            cam_id = group[4][0]
            angle = group[5][0]
            img_name = group[2][0].split("/")[-1]
            imgs = [cv2.imread(str(self.DATA_PATH / p)) / 255 for p in group[2]]
            imgs_mean = np.mean(imgs, axis=0)
            imgs_delta = np.sum([np.abs(img - imgs_mean) for img in imgs], axis=0)
            pth = str(self.BACKGROUND_DATA_PATH / trainval / is_empty / season / cam_id / angle / img_name)
            pth2 = str(self.MEAN_DATA_PATH / trainval / is_empty / season / cam_id / angle / img_name)

            cv2.imwrite(pth, np.clip(imgs_delta * 255, 0, 255).astype('uint8'))
            cv2.imwrite(pth2, np.clip(imgs_mean * 255, 0, 255).astype('uint8'))
        except:
            print("bad", group[2][0].split("/")[-1])
            pass

    def preprocess(self):

        train_metadata = pd.read_csv(self.DATA_PATH / "train_metadata.csv", parse_dates=['datetime'])
        train_metadata['season'] = train_metadata.seq_id.map(lambda x: x.split('#')[0])
        train_metadata['cam_id'] = train_metadata.seq_id.map(lambda x: x.split('#')[1])
        train_metadata['angle_id'] = train_metadata.seq_id.map(lambda x: x.split('#')[2])
        # train_metadata=train_metadata[train_metadata.season=='SER_S1']
        train_labels = pd.read_csv(self.DATA_PATH / "train_labels.csv", index_col="seq_id", usecols=['seq_id', 'empty'])

        # create dirs
        os.mkdir(str(self.BACKGROUND_DATA_PATH))
        os.mkdir(str(self.MEAN_DATA_PATH))
        for NEW_DATA_PATH in [self.BACKGROUND_DATA_PATH, self.MEAN_DATA_PATH]:
            os.mkdir(str(NEW_DATA_PATH / "train"))
            os.mkdir(str(NEW_DATA_PATH / "val"))
            for xx in ["empty", "animal"]:
                os.mkdir(str(NEW_DATA_PATH / "train" / xx))
                os.mkdir(str(NEW_DATA_PATH / "val" / xx))
                for s in set(train_metadata['season'].values):
                    trainval = "val" if s[-1] in ["9", "0"] else "train"
                    os.mkdir(str(NEW_DATA_PATH / trainval / xx / s))
                    for cam_id in set(train_metadata['cam_id'].values):
                        os.mkdir(str(NEW_DATA_PATH / trainval / xx / s / cam_id))
                        for angle_id in set(train_metadata['angle_id'].values):
                            os.mkdir(str(NEW_DATA_PATH / trainval / xx / s / cam_id / angle_id))

        train_metadata = train_metadata.set_index('seq_id')
        train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

        # remove sequences with one image
        train_metadata = train_metadata[train_metadata.groupby('seq_id')['file_name'].transform('count') > 1]

        train_gen_df = train_labels.join(train_metadata, how='right')

        def exctract_columns(df):
            return (str(df.index[0]),
                    df['empty'].values,
                    df['file_name'].values,
                    df['season'].values,
                    df['cam_id'].values,
                    df['angle_id'].values)

        groups = train_gen_df.groupby('seq_id').apply(exctract_columns).values

        pool = multiprocessing.Pool(self.processors)
        pool.map(self.process, groups)


if __name__ == "__main__":
    PreProcessor(sys.argv[1]).preprocess()
