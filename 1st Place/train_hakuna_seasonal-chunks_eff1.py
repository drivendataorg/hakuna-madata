from assets.utils import *
from config import config

# torch.cuda.set_device(0)

path = config.DATA_PATH# Change if you have the imagery mounted at a different location
train_metadata = pd.read_csv(path+"train_metadata_with_labels.csv")
train_metadata.index=train_metadata.seq_id
train_labels = pd.read_csv(path+"train_labels.csv", index_col="seq_id")
chunks = pd.read_csv(path+"train_in_chunks_per_season.csv", index_col="seq_id")

SZ=(384,512)
# SZ=(192,256)

src = (ImageList.from_df(path=path, df=train_metadata, cols="file_name")
       .split_none()
       .label_from_df(cols='labels', label_delim=';')
      )
data = (src.transform(get_transforms(max_rotate=5,max_warp=0, max_zoom=1.02,
                                     p_affine=.0 , p_lighting=.0,), size=SZ) #512x384
        .databunch(bs=16)
        .normalize(imagenet_stats))

acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

model = get_efficientnet("b1")
learn = Learner(data, model, wd=1e-2,model_dir="assets/models",
                   bn_wd=False, true_wd=True,
                    metrics=[acc_02, f_score],
               )
learn.unfreeze()

def train_on_chunk(path, learn, chunk, train_metadata):
    """
    This function enables learning only on a subset of the train set which we refer as chunk. 
    Chunks are conmprised of mostly one season but could have two or more.
    """
    subset = train_metadata.drop(chunks[chunks["chunk"]!=chunk].index).copy()
    
    # The line bellow makes sure everytime we create databunch it will have all 54 classes. 
    # The noise we introduce this way does not humper learning
    subset.labels[-1] = "aardvark;aardwolf;baboon;bat;batearedfox;buffalo;bushbuck;caracal;cattle;cheetah;civet;dikdik;duiker;eland;elephant;empty;gazellegrants;gazellethomsons;genet;giraffe;guineafowl;hare;hartebeest;hippopotamus;honeybadger;hyenaspotted;hyenastriped;impala;insectspider;jackal;koribustard;leopard;lionfemale;lionmale;mongoose;monkeyvervet;ostrich;otherbird;porcupine;reedbuck;reptiles;rhinoceros;rodents;secretarybird;serval;steenbok;topi;vulture;warthog;waterbuck;wildcat;wildebeest;zebra;zorilla"
    
    src = (ImageList.from_df(path=path, df=subset, cols="file_name")
           .split_none()
           .label_from_df(cols='labels', label_delim=';')
          )
    data = (src.transform(get_transforms(max_rotate=5,max_warp=0, max_zoom=1.02,
                                         p_affine=.0 , p_lighting=.0,), size=SZ) #512x384
            .databunch(bs=16)
            .normalize(imagenet_stats))
    learn.data=data
    return learn

# chunk 1 - Season 7

chunk=1
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 3e-5,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=10, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,2)
                           ])

# chunk 2 - Seasons 3, 4 and 51

chunk=2
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 1e-5,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=3, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )

# chunk 3 - Seasons 1, 2 and 6

chunk=3
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 0.8e-5,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=3, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )

# chunk 4 - Seasons 5

chunk=4
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.3e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ])

# learn.save("model_b1")

# chunk 5 - Seasons 9

chunk=5
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 0.2e-5,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=2, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )

# chunk 6 - Seasons 8

chunk=6
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.1e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ] )

# chunk 7- train further with the same season

chunk=7
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.1e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ] )

learn = train_on_chunk(path, learn, chunk, train_labels)
learn.fit(1, 0.05e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,15)
                           ]            #accumulate weights every 4 epochs to make effective batch size bigger 16x4=64
         )

# chunk 8 - Seasons 10

chunk=8
learn = train_on_chunk(path, learn, chunk, train_labels)

learn.fit(1, 0.1e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ])

learn.save("model_b1_season10")