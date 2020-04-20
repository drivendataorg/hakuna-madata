from datetime import datetime
import multiprocessing
from pathlib import Path
import logging
import logging.handlers
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as insres_preprocess_input
import numpy as np
from PIL import ImageFile
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from copy import deepcopy
import lightgbm as lgb
from PIL import Image
import os


ASSET_PATH = str(Path(__file__).parents[0] / "assets")
DATA_PATH = str(Path(__file__).parents[0] / "data")
IMGS_PATH = DATA_PATH
DF_PATH = DATA_PATH


logging.basicConfig(level=logging.INFO)
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_480_360_path = ASSET_PATH + "/" + 'insres_480_360.h5'
model_480_360_v2_path = ASSET_PATH + "/" + 'insres_480_360_v2.h5'
model_512_384_v2_path = ASSET_PATH + "/" + 'insres_512_384_v2.h5'
model_back_path = ASSET_PATH + "/" + 'background_insres_all_lr.h5'
model_mean_back_2models_path = ASSET_PATH + "/" + 'connected_model.h5'

lgb_models_names = ['empty', 'wildebeest', 'zebra', 'gazellethomsons', 'buffalo',
                    'elephant', 'hartebeest', 'impala', 'gazellegrants', 'giraffe',
                    'warthog', 'guineafowl', 'otherbird']

lgb_need_columns = [0, 1, 2, 4, 5, 6, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                    34, 35, 36, 37, 38, 39, 42, 43, 44, 46, 48, 49, 51, 52]

def add_values(line, values, img_n, imgs_count):
    v = np.array(values)[lgb_need_columns]
    if imgs_count > 2:
        line.extend(v)
    elif imgs_count == 2 and img_n < 3:
        line.extend(v)
    elif imgs_count == 1 and img_n == 1:
        line.extend(v)
    else:
        line.extend([-0.01 for _ in range(len(lgb_need_columns))])

def perform_inference():

    logging.info("loading csvs")
    test_metadata = pd.read_csv(DF_PATH + "/" + "test_metadata.csv", parse_dates=['datetime'])
    test_metadata['hour'] = test_metadata.datetime.dt.hour
    test_metadata['month'] = test_metadata.datetime.dt.month
    submission_format = pd.read_csv(DF_PATH + "/" + "submission_format.csv", index_col=0)
    logging.info("loading csvs: done")

    logging.info("loading models")
    model_480_360 = load_model(model_480_360_path)
    model_480_360_v2 = load_model(model_480_360_v2_path)
    model_512_384_v2 = load_model(model_512_384_v2_path)
    model_back = load_model(model_back_path)
    model_mean_back_2models = load_model(model_mean_back_2models_path)
    lgb_models_s9 = {n: lgb.Booster(model_file=ASSET_PATH + "/lgbs_v11/" + n + ".txt") for n in lgb_models_names}
    lgb_models_s10 = {n: lgb.Booster(model_file=ASSET_PATH + "/lgbs_v12/" + n + ".txt") for n in lgb_models_names}

    logging.info("loading models: done")

    logging.info("prepare submission dataframe")
    test_metadata = test_metadata.sort_values("file_name")
    test_seqs = test_metadata.seq_id.unique()
    logging.info('seqs len = ' + str(len(test_seqs)))
    output = np.zeros((len(test_seqs), submission_format.shape[1]))
    empty_sumbission = pd.DataFrame(
        np.stack(output),
        index=test_seqs,
        columns=submission_format.columns,
    )

    logging.info('empty_sumbission shape' + str(empty_sumbission.shape))
    logging.info("prepare submission dataframe: done")

    # if image is lost, lets set mean values
    empty_sumbission['aardvark'] = 0.000310
    empty_sumbission['aardwolf'] = 0.000171
    empty_sumbission['baboon'] = 0.001154
    empty_sumbission['bat'] = 0.000002
    empty_sumbission['batearedfox'] = 0.000245
    empty_sumbission['buffalo'] = 0.009228
    empty_sumbission['bushbuck'] = 0.000160
    empty_sumbission['caracal'] = 0.000047
    empty_sumbission['cattle'] = 0.000009
    empty_sumbission['cheetah'] = 0.000888
    empty_sumbission['civet'] = 0.000030
    empty_sumbission['dikdik'] = 0.001125
    empty_sumbission['duiker'] = 0.000032
    empty_sumbission['eland'] = 0.002281
    empty_sumbission['elephant'] = 0.008200
    empty_sumbission['empty'] = 0.759851
    empty_sumbission['gazellegrants'] = 0.006394
    empty_sumbission['gazellethomsons'] = 0.037591
    empty_sumbission['genet'] = 0.000028
    empty_sumbission['giraffe'] = 0.006195
    empty_sumbission['guineafowl'] = 0.004460
    empty_sumbission['hare'] = 0.000364
    empty_sumbission['hartebeest'] = 0.008055
    empty_sumbission['hippopotamus'] = 0.002017
    empty_sumbission['honeybadger'] = 0.000033
    empty_sumbission['hyenaspotted'] = 0.004108
    empty_sumbission['hyenastriped'] = 0.000089
    empty_sumbission['impala'] = 0.006635
    empty_sumbission['insectspider'] = 0.000304
    empty_sumbission['jackal'] = 0.000531
    empty_sumbission['koribustard'] = 0.000506
    empty_sumbission['leopard'] = 0.000158
    empty_sumbission['lionfemale'] = 0.002366
    empty_sumbission['lionmale'] = 0.000710
    empty_sumbission['mongoose'] = 0.000230
    empty_sumbission['monkeyvervet'] = 0.000235
    empty_sumbission['ostrich'] = 0.000509
    empty_sumbission['otherbird'] = 0.004275
    empty_sumbission['porcupine'] = 0.000231
    empty_sumbission['reedbuck'] = 0.001916
    empty_sumbission['reptiles'] = 0.000062
    empty_sumbission['rhinoceros'] = 0.000027
    empty_sumbission['rodents'] = 0.000040
    empty_sumbission['secretarybird'] = 0.000695
    empty_sumbission['serval'] = 0.000551
    empty_sumbission['steenbok'] = 0.000004
    empty_sumbission['topi'] = 0.001663
    empty_sumbission['vulture'] = 0.000033
    empty_sumbission['warthog'] = 0.005640
    empty_sumbission['waterbuck'] = 0.000232
    empty_sumbission['wildcat'] = 0.000039
    empty_sumbission['wildebeest'] = 0.076248
    empty_sumbission['zebra'] = 0.053718
    empty_sumbission['zorilla'] = 0.000016

    logging.info('groupping dataframe')
    groups = test_metadata.groupby('seq_id').apply(
        lambda df: (str(df.seq_id.values[0]), df['file_name'].values, df['hour'].values, df['month'].values)).values
    logging.info('groups count = ' + str(len(groups)))
    logging.info('groupping dataframe:done')

    seq_id_buf = []


    first_img_buf_480 = []
    second_img_buf_480 = []
    third_img_buf_480 = []
    first_img_buf_512 = []
    second_img_buf_512 = []
    third_img_buf_512 = []
    imgs_back_buf_512 = []
    imgs_mean_buf_512 = []
    imgs_back_buf_480 = []
    imgs_mean_buf_480 = []
    count_imgs = []

    hours = []
    months = []
    counter = 0
    batch_size = 32
    pool = multiprocessing.Pool(3)
    last_seq_id = groups[-1][0]

    imgs_reaging_time = datetime.now() - datetime.now()
    imgs_preprocess_time = datetime.now() - datetime.now()
    for a in groups:
        inference_start = datetime.now()
        seq_id = a[0]

        imgs = pool.map(read, [str(IMGS_PATH + "/" + p) for p in a[1][:3]])

        imgs_reaging_time += datetime.now() - inference_start
        imgs = np.array([i for i in imgs if i is not None])
        if len(imgs) > 0:


            # remove exif
            if imgs[0].shape[0] > 450 and np.mean(imgs[0][462:, 130:320]) > 250:
                imgs[:, 458:] = 0
            if imgs[0].shape[0] > 450 and np.median(imgs[0][462:, :100]) > 200 and np.median(imgs[0][462:, :100]) < 221:
                imgs[:, 458:, :120] = 0
                imgs[:, 458:, 460:] = 0
            elif imgs[0].shape[0] > 450 and np.median(imgs[0][462:, :60]) > 200 and np.median(imgs[0][462:, :60]) < 221:
                imgs[:, 458:, :70] = 0
                imgs[:, 458:, 460:] = 0
            if np.mean(imgs[0][360:, 130:320]) > 250:
                imgs[:, 355:] = 0
            if np.median(imgs[0][365:, :100]) > 200 and np.median(imgs[0][365:, :100]) < 221:
                imgs[:, 363:, :110] = 0
                imgs[:, 363:, 330:] = 0
            elif np.median(imgs[0][365:, :60]) > 200 and np.median(imgs[0][365:, :60]) < 221:
                imgs[:, 363:, :70] = 0
                imgs[:, 363:, 330:] = 0

            # cv2.imshow('', imgs[0])
            # cv2.waitKey(0)

            hours.append(a[2][0])
            months.append(a[3][0])

            imgs_480_360 = pool.map(cv2_resize, [(img, 480, 360) for img in imgs])
            imgs_512_384 = pool.map(cv2_resize, [(img, 512, 384) for img in imgs])

            first_img_buf_480.append(insres_preprocess_input(deepcopy(imgs_480_360[0])))
            first_img_buf_512.append(insres_preprocess_input(deepcopy(imgs_512_384[0])))
            if len(imgs_480_360) > 1:
                second_img_buf_480.append(insres_preprocess_input(deepcopy(imgs_480_360[1])))
                second_img_buf_512.append(insres_preprocess_input(deepcopy(imgs_512_384[1])))
            else:
                second_img_buf_480.append(np.zeros((360, 480, 3)))
                second_img_buf_512.append(np.zeros((384, 512, 3)))
            if len(imgs_480_360) > 2:
                third_img_buf_480.append(insres_preprocess_input(deepcopy(imgs_480_360[2])))
                third_img_buf_512.append(insres_preprocess_input(deepcopy(imgs_512_384[2])))
            else:
                third_img_buf_480.append(np.zeros((360, 480, 3)))
                third_img_buf_512.append(np.zeros((384, 512, 3)))

            imgs_mean = np.mean(np.asarray(imgs), axis=0)
            imgs_back = np.sum([np.abs(img - imgs_mean) for img in imgs], axis=0)
            imgs_back = np.clip(imgs_back, 0, 255)

            imgs_mean_buf_512.append(insres_preprocess_input(cv2.resize(imgs_mean, (512, 384))))
            imgs_mean_buf_480.append(insres_preprocess_input(cv2.resize(imgs_mean, (480, 360))))
            imgs_back_buf_512.append(cv2.resize(imgs_back, (512, 384)) / 255)
            imgs_back_buf_480.append(cv2.resize(deepcopy(imgs_back), (480, 360)) / 255)

            seq_id_buf.append(seq_id)

            count_imgs.append(len(imgs))

            imgs_preprocess_time += (datetime.now() - inference_start)
        else:
            logging.info(seq_id + " not found")

        if len(seq_id_buf) == batch_size or (seq_id == last_seq_id and len(seq_id_buf) > 0):
            counter += len(seq_id_buf)
            logging.info(counter)


            pred_back_as480 = model_back.predict(np.array(imgs_back_buf_480))
            pred_back_as512 = model_back.predict(np.array(imgs_back_buf_512))
            pred_mean_back_2models_as480 = model_mean_back_2models.predict([np.array(imgs_back_buf_480), np.array(imgs_mean_buf_480)])
            pred_mean_back_2models_as512 = model_mean_back_2models.predict([np.array(imgs_back_buf_512), np.array(imgs_mean_buf_512)])

            pred_model_480_360_as480_img1 = (model_480_360.predict(np.array(first_img_buf_480))+model_480_360.predict(np.array(first_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_as480_img2 = (model_480_360.predict(np.array(second_img_buf_480))+model_480_360.predict(np.array(second_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_as480_img3 = (model_480_360.predict(np.array(third_img_buf_480))+model_480_360.predict(np.array(third_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_as512_img1 = (model_480_360.predict(np.array(first_img_buf_512))+model_480_360.predict(np.array(first_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_480_360_as512_img2 = (model_480_360.predict(np.array(second_img_buf_512))+model_480_360.predict(np.array(second_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_480_360_as512_img3 = (model_480_360.predict(np.array(third_img_buf_512))+model_480_360.predict(np.array(third_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as480_img1 = (model_480_360_v2.predict(np.array(first_img_buf_480))+model_480_360_v2.predict(np.array(first_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as480_img2 = (model_480_360_v2.predict(np.array(second_img_buf_480))+model_480_360_v2.predict(np.array(second_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as480_img3 = (model_480_360_v2.predict(np.array(third_img_buf_480))+model_480_360_v2.predict(np.array(third_img_buf_480)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as512_img1 = (model_480_360_v2.predict(np.array(first_img_buf_512))+model_480_360_v2.predict(np.array(first_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as512_img2 = (model_480_360_v2.predict(np.array(second_img_buf_512))+model_480_360_v2.predict(np.array(second_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_480_360_v2_as512_img3 = (model_480_360_v2.predict(np.array(third_img_buf_512))+model_480_360_v2.predict(np.array(third_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_512_384_v2_as512_img1 = (model_512_384_v2.predict(np.array(first_img_buf_512))+model_512_384_v2.predict(np.array(first_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_512_384_v2_as512_img2 = (model_512_384_v2.predict(np.array(second_img_buf_512))+model_512_384_v2.predict(np.array(second_img_buf_512)[:, :, ::-1, :]))/2
            pred_model_512_384_v2_as512_img3 = (model_512_384_v2.predict(np.array(third_img_buf_512))+model_512_384_v2.predict(np.array(third_img_buf_512)[:, :, ::-1, :]))/2


            X_test = []
            for i in range(len(seq_id_buf)):
                pred_back_as480_i = pred_back_as480[i]
                pred_back_as512_i = pred_back_as512[i]
                pred_mean_back_2models_as480_i = pred_mean_back_2models_as480[i]
                pred_mean_back_2models_as512_i = pred_mean_back_2models_as512[i]
                pred_model_480_360_as480_img1_i = pred_model_480_360_as480_img1[i]
                pred_model_480_360_as480_img2_i = pred_model_480_360_as480_img2[i]
                pred_model_480_360_as480_img3_i = pred_model_480_360_as480_img3[i]
                pred_model_480_360_as512_img1_i = pred_model_480_360_as512_img1[i]
                pred_model_480_360_as512_img2_i = pred_model_480_360_as512_img2[i]
                pred_model_480_360_as512_img3_i = pred_model_480_360_as512_img3[i]
                pred_model_480_360_v2_as480_img1_i = pred_model_480_360_v2_as480_img1[i]
                pred_model_480_360_v2_as480_img2_i = pred_model_480_360_v2_as480_img2[i]
                pred_model_480_360_v2_as480_img3_i = pred_model_480_360_v2_as480_img3[i]
                pred_model_480_360_v2_as512_img1_i = pred_model_480_360_v2_as512_img1[i]
                pred_model_480_360_v2_as512_img2_i = pred_model_480_360_v2_as512_img2[i]
                pred_model_480_360_v2_as512_img3_i = pred_model_480_360_v2_as512_img3[i]
                pred_model_512_384_v2_as512_img1_i = pred_model_512_384_v2_as512_img1[i]
                pred_model_512_384_v2_as512_img2_i = pred_model_512_384_v2_as512_img2[i]
                pred_model_512_384_v2_as512_img3_i = pred_model_512_384_v2_as512_img3[i]


                count_img = count_imgs[i]
                hour = hours[i]
                month = months[i]

                median_img1 = np.median([pred_model_480_360_as480_img1_i,
                                         pred_model_480_360_as512_img1_i,
                                         pred_model_480_360_v2_as480_img1_i,
                                         pred_model_480_360_v2_as512_img1_i,
                                         pred_model_512_384_v2_as512_img1_i], axis=0)
                median_img2 = np.median([pred_model_480_360_as480_img2_i,
                                         pred_model_480_360_as512_img2_i,
                                         pred_model_480_360_v2_as480_img2_i,
                                         pred_model_480_360_v2_as512_img2_i,
                                         pred_model_512_384_v2_as512_img2_i], axis=0)
                median_img3 = np.median([pred_model_480_360_as480_img3_i,
                                         pred_model_480_360_as512_img3_i,
                                         pred_model_480_360_v2_as480_img3_i,
                                         pred_model_480_360_v2_as512_img3_i,
                                         pred_model_512_384_v2_as512_img3_i], axis=0)

                if count_img == 2:
                    mean_img = (median_img1 + median_img2 * 0.8) / 1.8
                    median_img3 = deepcopy(median_img1)
                elif count_img > 2:
                    mean_img = (median_img1 + median_img2 * 0.8 + median_img3 * 0.5) / 2.3
                else:
                    mean_img = median_img1
                    median_img3 = deepcopy(median_img1)
                    median_img2 = deepcopy(median_img1)

                empty_sumbission.loc[seq_id_buf[i]] = mean_img


                if count_img >1:
                    mean_back = np.median(
                        [pred_back_as480_i, pred_back_as512_i], axis=0)
                    mean_combiners = np.median([pred_mean_back_2models_as480_i, pred_mean_back_2models_as512_i], axis=0)
                else:
                    mean_back = deepcopy(mean_img)
                    mean_combiners = deepcopy(mean_img)


                line = []
                add_values(line, pred_model_480_360_as480_img1_i, img_n=1, imgs_count=count_img)
                add_values(line, pred_model_480_360_as480_img2_i, img_n=2, imgs_count=count_img)
                add_values(line, pred_model_480_360_as480_img3_i, img_n=3, imgs_count=count_img)
                add_values(line, pred_model_480_360_as512_img1_i, img_n=1, imgs_count=count_img)
                add_values(line, pred_model_480_360_as512_img2_i, img_n=2, imgs_count=count_img)
                add_values(line, pred_model_480_360_as512_img3_i, img_n=3, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as480_img1_i, img_n=1, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as480_img2_i, img_n=2, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as480_img3_i, img_n=3, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as512_img1_i, img_n=1, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as512_img2_i, img_n=2, imgs_count=count_img)
                add_values(line, pred_model_480_360_v2_as512_img3_i, img_n=3, imgs_count=count_img)
                add_values(line, pred_model_512_384_v2_as512_img1_i, img_n=1, imgs_count=count_img)
                add_values(line, pred_model_512_384_v2_as512_img2_i, img_n=2, imgs_count=count_img)
                add_values(line, pred_model_512_384_v2_as512_img3_i, img_n=3, imgs_count=count_img)
                add_values(line, mean_back, img_n=1, imgs_count=count_img)
                add_values(line, median_img1, img_n=1, imgs_count=count_img)
                add_values(line, median_img2, img_n=1, imgs_count=count_img)
                add_values(line, median_img3, img_n=1, imgs_count=count_img)
                add_values(line, mean_combiners, img_n=1, imgs_count=count_img)
                line.append(hour)
                line.append(month)
                X_test.append(np.array(line))

            for name in lgb_models_names:
                lgb_model_s9 = lgb_models_s9[name]
                lgb_model_s10 = lgb_models_s10[name]
                lgb_pred_s9 = lgb_model_s9.predict(np.array(X_test))
                lgb_pred_s10 = lgb_model_s10.predict(np.array(X_test))

                lgb_pred = (lgb_pred_s9+lgb_pred_s10)/2
                for seq_id, lgb_p in zip(seq_id_buf, lgb_pred):
                    empty_sumbission.loc[seq_id, name] = lgb_p


            seq_id_buf = []
            first_img_buf_480 = []
            second_img_buf_480 = []
            third_img_buf_480 = []
            first_img_buf_512 = []
            second_img_buf_512 = []
            third_img_buf_512 = []
            imgs_back_buf_512 = []
            imgs_mean_buf_512 = []
            imgs_back_buf_480 = []
            imgs_mean_buf_480 = []
            count_imgs = []
            hours = []
            months = []

            inference_stop = datetime.now()
            logging.info(f"NN Took {inference_stop - inference_start}.")
            logging.info(f"reading {imgs_reaging_time}.")
            logging.info(f"preprocess {imgs_preprocess_time - imgs_reaging_time}.")

            imgs_reaging_time = datetime.now() - datetime.now()
            imgs_preprocess_time = datetime.now() - datetime.now()


    empty_sumbission.astype(np.float).to_csv("submission.csv", index=True)



if __name__ == "__main__":
    def cv2_resize(img_and_size):
        return cv2.resize(img_and_size[0], (img_and_size[1], img_and_size[2]))


    def read(path):
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            s1 = img.size[0] // 4
            s2 = img.size[1] // 4
            if s1 < 480 or s2 < 360:
                img = img.resize((512, 384), Image.ANTIALIAS)
            else:
                img = img.resize((s1, s2), Image.ANTIALIAS)
            return np.asarray(img)
        except:
            return None


    perform_inference()
