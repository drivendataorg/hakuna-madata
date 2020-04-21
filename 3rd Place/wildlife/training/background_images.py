import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.models import load_model
import sys

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


from tensorflow.keras.applications import inception_resnet_v2

DATA_PATH = Path(sys.argv[1])

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

val_gen_df['file_name'] = val_gen_df.apply(
    lambda x: str(DATA_PATH) + '/preprocessed/background/val/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

train_gen_df['file_name'] = train_gen_df.apply(
    lambda x: str(DATA_PATH) + '/preprocessed/background/train/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

class Pipeline:
    def __init__(self,resolution,batch_size,epoches,start_lr):
        self.resolution = resolution
        self.batch_size = batch_size
        self.epoches = epoches
        self.start_lr = start_lr


pipelines = [

    Pipeline(resolution=(240, 320),
             batch_size=32,
             epoches=20,
             start_lr=0.0001
             ),
    Pipeline(resolution=(360, 480),
             batch_size=16,
             epoches=15,
             start_lr=0.00003
             ),
    Pipeline(resolution=(384, 512),
             batch_size=16,
             epoches=15,
             start_lr=0.00001
             )]

train_steps = 5000
val_steps = 100
evaluation_steps = 5000

version = sys.argv[2]
if version == "0":
    train_steps = 20
    val_steps = 10
    evaluation_steps = 200



datagen_flip = ImageDataGenerator(preprocessing_function=inception_resnet_v2.preprocess_input, horizontal_flip=True)
datagen = ImageDataGenerator(preprocessing_function=inception_resnet_v2.preprocess_input)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_gens(target_size, batch_size):
    train_datagen = datagen_flip.flow_from_dataframe(
        dataframe=train_gen_df,
        x_col="file_name",
        y_col=label_columns,
        class_mode="other",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_datagen = datagen.flow_from_dataframe(
        dataframe=val_gen_df,
        x_col="file_name",
        y_col=label_columns,
        class_mode="other",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )
    return train_datagen, val_datagen


model = load_model("./insres_360_v2.h5")

def get_scheduler(start_lr):
    def scheduler(epoch):
        lr = start_lr * np.exp(0.15 * (- epoch))
        print("lr =", lr)
        return lr
    return scheduler

for pipeline in pipelines:
    callback = LearningRateScheduler(get_scheduler(pipeline.start_lr))

    train_datagen, val_datagen = get_gens(pipeline.resolution, pipeline.batch_size)

    print("resolution {} all layers".format(pipeline.resolution))

    opt = Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
    model.fit_generator(
        train_datagen,
        steps_per_epoch=train_steps,
        validation_data=val_datagen,
        validation_steps=val_steps,
        callbacks=[callback],
        epochs=pipeline.epoches
    )
    model_name = "background_insres_{}_v{}".format(pipeline.resolution[0], version)   + ".h5"
    model.save(model_name)
    print(model_name, 'evaluation')
    model.evaluate(val_datagen, steps=evaluation_steps)

