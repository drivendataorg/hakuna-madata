import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.callbacks import LearningRateScheduler

import sys
import random
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, Input, Lambda,Embedding,Concatenate,Flatten, Conv2D

from tensorflow.keras.models import load_model
model_mean = load_model("insres_360_v2.h5", compile=False)
model_back = load_model("background_insres_360_v1.h5", compile=False)

DATA_PATH = Path(sys.argv[1])

MEAN_DATASET_PATH = DATA_PATH / 'preprocessed' / 'mean'
BACK_DATASET_PATH = DATA_PATH / 'preprocessed' / 'background'

train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata['season'] = train_metadata.seq_id.map(lambda x: x.split('#')[0])
train_metadata = train_metadata.sort_values('file_name').set_index('seq_id')

train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

count = train_metadata.groupby('seq_id').size().reset_index().set_index('seq_id')
count.columns = ['seq_id_count']
train_metadata = train_metadata.join(count, on='seq_id', how='left')

train_metadata = train_metadata[train_metadata.seq_id_count > 1]
train_metadata=train_metadata.sort_values('file_name').groupby('seq_id').first()

train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])
train_metadata['cam_id'] = train_metadata.index.map(lambda x: x.split('#')[1])
train_metadata['angle_id'] = train_metadata.index.map(lambda x: x.split('#')[2])

train_seasons = ['SER_S1','SER_S2', 'SER_S3', 'SER_S4', 'SER_S5', 'SER_S6', 'SER_S7', 'SER_S8']
val_seasons = ["SER_S9"]

val_x = train_metadata[train_metadata.season.isin(val_seasons)]
val_y = train_labels[train_labels.index.isin(val_x.index)]

train_metadata = train_metadata[train_metadata.season.isin(train_seasons)]
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

train_gen_df = train_labels.join(train_metadata)
val_gen_df = val_y.join(val_x)
label_columns = train_labels.columns.tolist()


val_gen_df['mean_file_name'] = val_gen_df.apply(
    lambda x: str(MEAN_DATASET_PATH) + "/"+ 'val/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

val_gen_df['back_file_name'] = val_gen_df.apply(
    lambda x: str(BACK_DATASET_PATH) + "/"+ 'val/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

train_gen_df['mean_file_name'] = train_gen_df.apply(
    lambda x: str(MEAN_DATASET_PATH) + "/"+ 'train/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

train_gen_df['back_file_name'] = train_gen_df.apply(
    lambda x: str(BACK_DATASET_PATH) + "/"+ 'train/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

target_size = (360, 480)
batch_size = 16

from tensorflow.keras.applications import inception_resnet_v2

datagen = ImageDataGenerator()

train_datagen_x = datagen.flow_from_dataframe(
    dataframe=train_gen_df,
    x_col="mean_file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True
)
val_datagen_x = datagen.flow_from_dataframe(
    dataframe=val_gen_df,
    x_col="mean_file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True
)

def back_mean_gen(df, flip=True):
    while True:
        sample = df.sample(n = 8)
        mean_imgs = np.array([inception_resnet_v2.preprocess_input(cv2.resize(cv2.imread(p), (target_size[1], target_size[0]))[:,:,::-1]) for p in sample.mean_file_name])
        back_imgs = np.array([inception_resnet_v2.preprocess_input(cv2.resize(cv2.imread(p), (target_size[1], target_size[0]))[:,:,::-1]) for p in sample.back_file_name])
        if flip and random.choice([True, False]):
            mean_imgs=mean_imgs[:,:,::-1,:]
            back_imgs=back_imgs[:,:,::-1,:]
        labels = sample[label_columns].values
        yield ((back_imgs, mean_imgs), labels)

val_datagen = back_mean_gen(val_gen_df[val_gen_df.mean_file_name.isin(val_datagen_x.filenames)], flip=False)
train_datagen = back_mean_gen(train_gen_df[train_gen_df.mean_file_name.isin(train_datagen_x.filenames)])



for layer in model_mean.layers:
    layer._name = str('mean_') + layer.name
for layer in model_back.layers:
    layer._name = str('back_') + layer.name


back_input = model_back.layers[-4].output
mean_input = model_mean.layers[-4].output

for layer in model_back.layers:
    layer.trainable = False

for layer in model_back.layers[-100:]:
    layer.trainable = True

for layer in model_mean.layers:
    layer.trainable = False

for layer in model_mean.layers[-100:]:
    layer.trainable = True

x = Concatenate(axis=3)([back_input, mean_input])
x = Conv2D(filters=1024, kernel_size=3)(x)
x = GlobalMaxPooling2D()(x)
x = Dense(54, activation="sigmoid")(x)
model = Model(inputs=[model_back.input, model_mean.input], outputs=x)
model.summary()


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy


rms = Adam(learning_rate=0.0003)

model.compile(optimizer=rms, loss="binary_crossentropy", metrics=[categorical_accuracy])

def scheduler(epoch):
    lr = 0.0002 * np.exp(0.1 * (- epoch))
    print('lr', lr)
    return lr


callback = LearningRateScheduler(scheduler)

version = sys.argv[2]
if version == '0':
    model.fit_generator(
        train_datagen,
        steps_per_epoch=100,
        validation_data=val_datagen,
        validation_steps=100,
        epochs=30,
        callbacks=[callback],
    )

    model.save("connected_model_v0.h5")
else:

    model.fit_generator(
        train_datagen,
        steps_per_epoch=5000,
        validation_data=val_datagen,
        validation_steps=100,
        epochs=30,
        callbacks=[callback],
    )

    model.save("connected_model_v1.h5")

    model.evaluate(val_datagen, steps=5000)