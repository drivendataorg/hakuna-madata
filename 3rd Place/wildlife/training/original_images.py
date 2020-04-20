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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout
import sys


from tensorflow.keras.applications import inception_resnet_v2
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

version = sys.argv[2]


class Pipeline:
    def __init__(self,resolution,batch_size,epoches,start_lr):
        self.resolution = resolution
        self.batch_size = batch_size
        self.epoches = epoches
        self.start_lr = start_lr


if version == '0':
    pipelines = [
        Pipeline(resolution=(224, 224),
                 batch_size=48,
                 epoches=20,
                 start_lr=0.0004
                 ),
        Pipeline(resolution=(299, 299),
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
    train_steps = 10
    val_steps = 10
    evaluation_steps = 100
elif version == '1':
    pipelines = [
        Pipeline(resolution=(240, 320),
                 batch_size=32,
                 epoches=20,
                 start_lr=0.0003
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
    train_steps = 10000
    val_steps = 100
    evaluation_steps = 5000
elif version == '2':
    pipelines = [
        Pipeline(resolution=(224, 224),
                 batch_size=48,
                 epoches=25,
                 start_lr=0.0004
                 ),
        Pipeline(resolution=(299, 299),
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
else:
    resolutions = []
    epoch_train_steps = 0
    epoch_val_steps = 0
    epoches_on_resolution = 0
    evaluation_steps = 0
    raise AttributeError('version expected 0 or 1 or 2')


DATA_PATH = Path(sys.argv[1])

train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata = train_metadata.sort_values('file_name').groupby('seq_id').first()
train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])

train_metadata['file_name'] = train_metadata.apply(
    lambda x: (DATA_PATH / x.file_name), axis=1
)

train_seasons = ['SER_S1', 'SER_S2', "SER_S3", 'SER_S4', 'SER_S5', 'SER_S6', 'SER_S7', 'SER_S8']
val_seasons = ["SER_S9"] # season 10 will be used on the next stage

# split out validation first
val_x = train_metadata[train_metadata.season.isin(val_seasons)]
val_y = train_labels[train_labels.index.isin(val_x.index)]

# reduce training
train_metadata = train_metadata[train_metadata.season.isin(train_seasons)]
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

train_gen_df = train_labels.join(train_metadata.file_name.apply(lambda path: str(path)))
val_gen_df = val_y.join(val_x.file_name.apply(lambda path: str(path)))
label_columns = train_labels.columns.tolist()

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

def get_scheduler(start_lr):
    def scheduler(epoch):
        lr = start_lr * np.exp(0.15 * (- epoch))
        print("lr =", lr)
        return lr
    return scheduler

transfer = inception_resnet_v2.InceptionResNetV2(include_top=False)
max_pooling = GlobalMaxPooling2D(name="max_pooling")(transfer.output)
drop_out = Dropout(0.05, name="dropout1")(max_pooling)
outputs = Dense(len(label_columns), activation="sigmoid")(drop_out)
model = Model(inputs=transfer.input, outputs=outputs)

for pipeline in pipelines:
    callback = LearningRateScheduler(get_scheduler(pipeline.start_lr))

    train_datagen, val_datagen = get_gens(pipeline.resolution, pipeline.batch_size)
    for layersN in [1, 100, 200]:
        print("resolution {} last {} layers".format(pipeline.resolution, layersN))
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-layersN:]:
            layer.trainable = True

        opt = Adam(0.00002)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
        model.fit_generator(
            train_datagen,
            steps_per_epoch=train_steps,
            validation_data=val_datagen,
            validation_steps=val_steps,
            epochs=1)

    for layer in model.layers:
        layer.trainable = True

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
    model_name = "insres_{}_v{}".format(pipeline.resolution[0], version)   + ".h5"
    model.save(model_name)
    print(model_name, 'evaluation')
    model.evaluate(val_datagen, steps=evaluation_steps)

