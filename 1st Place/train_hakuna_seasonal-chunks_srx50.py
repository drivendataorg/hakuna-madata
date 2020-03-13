#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from assets.utils import *
from config import config

# torch.cuda.set_device(1)


# In[ ]:


path = config.DATA_PATH# Change if you have the imagery mounted at a different location
train_metadata = pd.read_csv(path+"train_metadata_with_labels.csv")
train_metadata.index=train_metadata.seq_id
train_labels = pd.read_csv(path+"train_labels.csv", index_col="seq_id")
chunks = pd.read_csv(path+"train_in_chunks_per_season.csv", index_col="seq_id")


# In[ ]:


SZ=(384,512)
# SZ=(192,256)


# In[ ]:


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

learn = cnn_learner(data,
                    base_arch=get_srx50,
                    cut=-2, 
                    custom_head=Head(512*4,data.c, 0.0),
                    model_dir="assets/models",
                    bn_wd=False,
                    true_wd=True,
                    metrics=[acc_02, f_score]
               )

learn.unfreeze()


# In[ ]:


# train_metadata = pd.concat([train_metadata.loc[train_metadata.index.str.startswith("SER_S"+str(i))].sample(50) for i in range(1,11)],0)
# train_metadata


# In[ ]:


def train_on_chunk(path, learn, chunk, train_metadata):
    """
    This function enables learning only on a subset of the train set which we refer as chunk. 
    Chunks are conmprised of mostly one season but could have two or more.
    """
    s7 = train_metadata.drop(chunks[chunks["chunk"]!=chunk].index).copy()
    
    # The line bellow makes sure everytime we create databunch it will have all 54 classes. 
    # The noise we introduce this way does not humper learning
    s7.labels[-1] = "aardvark;aardwolf;baboon;bat;batearedfox;buffalo;bushbuck;caracal;cattle;cheetah;civet;dikdik;duiker;eland;elephant;empty;gazellegrants;gazellethomsons;genet;giraffe;guineafowl;hare;hartebeest;hippopotamus;honeybadger;hyenaspotted;hyenastriped;impala;insectspider;jackal;koribustard;leopard;lionfemale;lionmale;mongoose;monkeyvervet;ostrich;otherbird;porcupine;reedbuck;reptiles;rhinoceros;rodents;secretarybird;serval;steenbok;topi;vulture;warthog;waterbuck;wildcat;wildebeest;zebra;zorilla"
    
    src = (ImageList.from_df(path=path, df=s7, cols="file_name")
           .split_none()
           .label_from_df(cols='labels', label_delim=';')
          )
    data = (src.transform(get_transforms(max_rotate=5,max_warp=0, max_zoom=1.02,
                                         p_affine=.0 , p_lighting=.0,), size=SZ) #512x384
            .databunch(bs=16)
            .normalize(imagenet_stats))
    learn.data=data
    return learn


# # chunk 1 - Season 7

# In[ ]:


chunk=1
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 3e-5*3,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=10, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,2)
                           ])


# # chunk 2 - Seasons 3, 4 and 51

# In[ ]:


chunk=2
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 1e-5*3,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=3, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )


# # chunk 3 - Seasons 1, 2 and 6

# In[ ]:


chunk=3
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 0.8e-5*3,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=3, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )


# # chunk 4 - Seasons 5

# In[ ]:


chunk=4
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.3e-5*3,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ])

# learn.save("model_b1")


# # chunk 5 - Seasons 9

# In[ ]:


chunk=5
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit_one_cycle(1, 0.2e-5*2,
                    pct_start=0.0002, #first 500 epochs slowly increase LR
                    div_factor=2, # then anealying to LR/10
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,4)
                           ] )


# # chunk 6 - Seasons 8

# In[ ]:


chunk=6
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.1e-5*2,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ] )


# # chunk 7- train further with the same season

# In[ ]:


chunk=7
learn = train_on_chunk(path, learn, chunk, train_metadata)

learn.fit(1, 0.1e-5*2,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ] )


# In[ ]:


learn = train_on_chunk(path, learn, chunk, train_labels)
learn.fit(1, 0.05e-5*2,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,15)
                           ]            #accumulate weights every 4 epochs to make effective batch size bigger 16x4=64
         )


# In[ ]:


learn.save("best-thu")


# # chunk 8 - Seasons 10

# In[ ]:


chunk=8
learn = train_on_chunk(path, learn, chunk, train_labels)

learn.fit(1, 0.1e-5,
                           callbacks = [#BnFreeze(learn), 
                            AccumulateStep(learn,8)
                           ])


# In[ ]:


learn.save("best-sat-0075")


# In[ ]:




