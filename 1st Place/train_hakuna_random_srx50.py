from assets.utils import *
from config import config

path = config.DATA_PATH
train_metadata = pd.read_csv(path+"train_metadata_with_labels.csv")
train_labels = pd.read_csv(path+"train_labels.csv", index_col="seq_id")

src = (ImageList.from_df(path=path, df=train_metadata, cols="file_name")
       .split_none()
       .label_from_df(cols='labels', label_delim=';'))

data = (src.transform(get_transforms(max_rotate=5,max_warp=0, max_zoom=1.02,
                                     p_affine=.0 , p_lighting=.0,), size=(384,512)) 
                      .databunch(bs=13)
                      .normalize(imagenet_stats))

acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data,
                    base_arch=get_srx50, 
                    cut=-2, 
                    custom_head=utils.Head(512*4,data.c, 0.0),
                    model_dir="assets/models",
                    bn_wd=False, 
                    true_wd=True,
                    metrics=[acc_02, f_score])

learn.unfreeze()
learn.fit_one_cycle(1, 
                    1e-4,
                    pct_start=0.0002, #first ~500 epochs slowly increase LR
                    div_factor=100, # then aneal to LR/100
                    callbacks = [AccumulateStep(learn,3)])

learn.save("model_srx50")

