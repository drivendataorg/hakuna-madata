import pandas as pd
from config import config

path = config.DATA_PATH

train_metadata = pd.read_csv(path+"train_metadata.csv")
train_labels = pd.read_csv(path+"train_labels.csv", index_col="seq_id")

l = lambda x: ";".join(list(train_labels.loc[x].loc[train_labels.loc[x] > 0].index))
train_metadata['labels'] = train_metadata.seq_id.map(l)

train_metadata.to_csv(path+"train_metadata_with_labels.csv",index=False)
train_metadata.head(3)