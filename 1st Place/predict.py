from fastai.vision import *
from datetime import datetime
import logging
from assets.models import pretrainedmodels
from assets.models import efficientnets
from assets import utils
from config import config

path = config.DATA_PATH
# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

vision.data.open_image = utils.open_croped_image1


def perform_inference():
    """This is the main function executed at runtime in the cloud environment. """
    logging.info("Loading model.")
    
    logging.info("Loading and processing metadata.")
    # our preprocessing selects the first image for each sequence
    test_metadata = pd.read_csv(path+"test_metadata.csv", index_col="seq_id")
    print("total images",test_metadata.shape)
    test_metadata["sid"]=test_metadata.index
    test_metadata["seq_id"]=test_metadata.index
    
#     filename=test_metadata.groupby("sid")["file_name"].apply(';'.join)
#     test_metadata=test_metadata.groupby("sid").first()
#     test_metadata["file_name2"] = filename

#     print("unique sequences",test_metadata.shape)

    print(test_metadata.head(2))
    test_metadata = (
        test_metadata.sort_values("file_name")#.groupby("seq_id").first().reset_index()
    )
    ################################################
    #test_metadata = test_metadata.sample(100)
    #################################################

    logging.info("Loading as databunch.")
    
#     learn = load_learner('assets/')
    df=pd.read_csv("assets/train.csv")
    src = (ImageList.from_df(path="",folder=path, df=df, cols="file_name").split_none()
       .label_from_df(cols='labels', label_delim=';'))



    # B3
    logging.info("B3 ...")
    vision.data.open_image = utils.open_croped_image1
    data = (src.transform(get_transforms(), size=(128*3,256*2)).databunch(bs=64).normalize(imagenet_stats))

    B="b3"
    model_name = 'efficientnet-'+B
    model = efficientnets.EfficientNet.from_name(model_name)
    model.add_module('_fc',nn.Linear(1536, 54))
    learn = Learner(data, model, wd=1e-2, bn_wd=False, true_wd=True,model_dir="assets/models")
    learn.data.add_test(ImageList.from_df(path=path, df=test_metadata, cols='file_name'))
    learn.data.batch_size = 32 # 
    learn.load("model_b3")
    p4, y = learn.get_preds(DatasetType.Test)



    # B1
    logging.info("B1 ...")
    vision.data.open_image = utils.open_croped_image1
    data = (src.transform(get_transforms(), size=(128*3,256*2)).databunch(bs=128).normalize(imagenet_stats))
    B="b1"
    model_name = 'efficientnet-'+B
    model = efficientnets.EfficientNet.from_name(model_name)
    model.add_module('_fc',nn.Linear(1280, 54))
    learn = Learner(data, model, wd=1e-2, bn_wd=False, true_wd=True,model_dir="assets/models")
    learn.load("model_b1_season10")
    learn.data.add_test(ImageList.from_df(path=path, df=test_metadata, cols='file_name'))
    learn.data.batch_size = 32 # 

    p5, y = learn.get_preds(DatasetType.Test)


    # seresnext50 3x
    vision.data.open_image = utils.open_croped_image1
    data = (src.transform(get_transforms(), size=(128*3,256*2)).databunch(bs=16).normalize(imagenet_stats))
    learn = cnn_learner(data, base_arch=get_srx50, cut=-2, custom_head=utils.Head(512*4,len(data.classes), 0.0),model_dir="assets/models") 
    learn.load("best-sat-0075")
    
    learn.data.add_test(ImageList.from_df(path=path, df=test_metadata, cols='file_name'))
    learn.data.batch_size = 32 # 

    logging.info("Starting inference.")
    logging.info("next ...")
    inference_start = datetime.now()
    p1, y = learn.get_preds(DatasetType.Test)

    logging.info("next ...")
    learn.load("best-thu")
    p2, y = learn.get_preds(DatasetType.Test)

    logging.info("next ...")
    learn.load("model_srx50")
    p3, y = learn.get_preds(DatasetType.Test)

    # flipped images
    logging.info("next ...")
    vision.data.open_image = utils.open_croped_image_flipped
    data = (src.transform(get_transforms(), size=(128*3,256*2)).databunch(bs=32).normalize(imagenet_stats))
    learn.data=data
    learn.load("model_srx50")
    p3a, y = learn.get_preds(DatasetType.Test)

    ##############
    # break the bone
    ######################################

    # PREDICTIONS AVG
    preds =p1*.05 + p2*.05 + p3*.4 +p3a*.4 + p4*.05 + p5*.05 
        

    inference_stop = datetime.now()
    logging.info(f"Inference complete. Took {inference_stop - inference_start}.")
    
    preds = pd.DataFrame(preds.numpy(), columns = ["f"+str(i+1) for i in range(54)])
    preds.index = test_metadata.seq_id
    
    # gmean works better for empty images
    gmean = preds.sort_values('seq_id').groupby('seq_id').apply(lambda group: group.product() ** (1 / float(len(group))))

    preds = preds.sort_values('seq_id').groupby('seq_id').mean()
    preds["f15"] = gmean["f15"]
    preds = preds.values

    logging.info("Setting up submission file.")
    submission_format = pd.read_csv(path+"submission_format.csv", index_col=0)
    # Check our predictions are in the same order as the submission format
    assert np.all(
        test_metadata.seq_id.unique().tolist() == submission_format.index.to_list()
    )



    for c in submission_format.columns:
        if c in learn.data.classes:
            idx = learn.data.classes.index(c)
            print(idx)
            submission_format[c] = preds[:, idx]


    # We want to ensure all of our data are floats, not integers
    my_submission = submission_format.astype(np.float)

    # Save out submission to root of directory
    my_submission.to_csv("submission.csv", index=True)
    logging.info(f"Submission saved.")
    
if __name__ == "__main__":
    perform_inference()
