import logging
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageFile
from scipy.stats import gmean
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

# This must be set to load some images using PIL, which Keras uses.
ImageFile.LOAD_TRUNCATED_IMAGES = True

ASSET_PATH = Path(__file__).parents[0] / "assets"
MODEL_PATH1 = ASSET_PATH / "rx50_w7_s4_e0.pth"
MODEL_PATH2 = ASSET_PATH / "rx101_w7_s3_e2.pth"  # ASSET_PATH / "rx50_v7_s4_e6.pth"
logging.info(MODEL_PATH1)
logging.info(MODEL_PATH2)

# The images will live in a folder called 'data' in the container
DATA_PATH = Path("/mnt/ssd1/dataset/wild")
if not osp.exists(DATA_PATH):
    DATA_PATH = Path(__file__).parents[0] / "/media/n01z3/ssd1_intel/dataset/wild"

CSV_FILENAME = "test_metadata.csv"
# if osp.exists(osp.join(DATA_PATH, "test_metadata.csv")):
#     CSV_FILENAME = "test_metadata.csv"

BATCH_SIZE = 1
IMG_SIZE = 360
SOFTMAX = True  # flag to apply softmax or sigmoid at logits

LABELS = [
    "aardvark",
    "aardwolf",
    "baboon",
    "bat",
    "batearedfox",
    "buffalo",
    "bushbuck",
    "caracal",
    "cattle",
    "cheetah",
    "civet",
    "dikdik",
    "duiker",
    "eland",
    "elephant",
    "empty",
    "gazellegrants",
    "gazellethomsons",
    "genet",
    "giraffe",
    "guineafowl",
    "hare",
    "hartebeest",
    "hippopotamus",
    "honeybadger",
    "hyenaspotted",
    "hyenastriped",
    "impala",
    "insectspider",
    "jackal",
    "koribustard",
    "leopard",
    "lionfemale",
    "lionmale",
    "mongoose",
    "monkeyvervet",
    "ostrich",
    "otherbird",
    "porcupine",
    "reedbuck",
    "reptiles",
    "rhinoceros",
    "rodents",
    "secretarybird",
    "serval",
    "steenbok",
    "topi",
    "vulture",
    "warthog",
    "waterbuck",
    "wildcat",
    "wildebeest",
    "zebra",
    "zorilla",
]


def get_model(weight):
    weight = str(weight)
    if "101" in weight:
        model = resnext101_32x8d(num_classes=54)
    else:
        model = resnext50_32x4d(num_classes=54)
    model = model.cuda()

    print(weight)
    checkpoint = torch.load(weight, map_location="cpu")
    if "state_dict" in checkpoint.keys():
        state_dict = checkpoint["state_dict"]
        sanitized = {}
        for k, v in state_dict.items():
            # if '101' in weight:
            #     sanitized[k.replace("model.model", "model")] = v
            # else:
            sanitized[k.replace("model.", "")] = v
            sanitized[k.replace("model.last_linear.", "fc.")] = v

        model.load_state_dict(sanitized, strict=False)
        if "101" in weight:
            save_name = "../ilya/assets/resnext101_w1_epoch4.pth"
        else:
            save_name = "../ilya/assets/resnext50_w8_epoch0.pth"
        torch.save(model.state_dict(), save_name)

    else:
        model.load_state_dict(checkpoint)

    del checkpoint
    model.eval()
    return model


class HakunaInferDataset:
    def __init__(self, mode, data_path, long_side=IMG_SIZE):
        assert mode in ["train", "val", "test"], f"unknown mode {mode}"
        self.path = data_path
        self.long_side = long_side
        self.mode = mode

        if self.mode == "test":
            # print(DATA_PATH)
            df_meta = pd.read_csv(DATA_PATH / "test_metadata.csv", index_col="seq_id")
            # print(df_meta.head())
            df_meta.sort_values("file_name", inplace=True)
            # self.path = "data"

        else:
            df_path = osp.join(self.path, "annotation/valid.csv")
            df_meta = pd.read_csv(df_path)

        self.groups = list(df_meta.groupby("seq_id"))

        if self.mode == "val":
            df_labels = pd.read_csv(osp.join(self.path, "annotation/train_labels.csv"))
            df_labels = df_labels[df_labels["seq_id"].isin(df_meta["seq_id"])]
            self.labels = df_labels[LABELS].values
            self.seq2index = dict([(seq, n) for n, seq in enumerate(df_labels["seq_id"])])

        self.test_metadata = df_meta.groupby("seq_id").first().reset_index()

    def image_to_tensor(self, img):
        np_img = np.asarray(img, dtype=np.uint8)
        np_img = np_img - np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
        np_img = np_img / np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
        np_img = np.rollaxis(np_img, 2)
        img_tensor = torch.from_numpy(np_img)
        return img_tensor

    def get_image(self, full_path):
        img = Image.open(full_path)
        img1 = img.resize((512, 384), Image.ANTIALIAS)

        # resize to 512 longest size:
        w, h = img1.size
        ratio = max(h / IMG_SIZE, w / IMG_SIZE)
        # want them to be divizable by 16
        new_w = int((w / ratio) // 16 * 16)
        new_h = int((h / ratio) // 16 * 16)
        img2 = img1.resize((new_w, new_h), resample=Image.LANCZOS)
        return self.image_to_tensor(img1), self.image_to_tensor(img2)

    def __getitem__(self, idx):
        batch = {}

        seq_id, group_df = self.groups[idx]
        batch["seq_id"] = seq_id
        images1, images2 = [], []
        for file_name in group_df["file_name"]:
            img1, img2 = self.get_image(osp.join(str(self.path), file_name))
            images1.append(img1)
            images2.append(img2)

        batch["images1"] = torch.stack(images1)
        batch["images2"] = torch.stack(images2)
        if self.mode == "val":
            batch["label"] = self.labels[self.seq2index.get(seq_id)]

        return batch

    def __len__(self):
        return len(self.groups)


class Loss(_Loss):
    """Loss which supports addition and multiplication"""

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return WeightedLoss(self, value)
        else:
            raise ValueError("Loss should be multiplied by int or float")

    def __rmul__(self, other):
        return self.__mul__(other)


class WeightedLoss(Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = torch.Tensor([weight])

    def forward(self, *inputs):
        l = self.loss(*inputs)
        self.weight = self.weight.to(l.device)
        return l * self.weight[0]


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)


class MultiLabelSoftMarginLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, y_pred, y_true):
        if len(y_true.shape) != 1:
            y_true_one_hot = y_true.float()
        else:
            num_classes = y_pred.size(1)
            y_true_one_hot = torch.zeros(y_true.size(0), num_classes, dtype=torch.float, device=y_pred.device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1.0)

        return self.loss(y_pred, y_true_one_hot)


def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def perform_inference():
    """This is the main function executed at runtime in the cloud environment. """
    logging.info("Loading model.")

    models = []
    for path in [MODEL_PATH1, MODEL_PATH2]:
        # models.append(torch.jit.load(str(path)).cuda())
        models.append(get_model(path))
        logging.info(f"Loading and processing metadata. {path}")

    # Instantiate test data loader
    test_dataset = HakunaInferDataset(mode="test", data_path=DATA_PATH)
    test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=BATCH_SIZE)

    logging.info("Starting inference.")

    # Preallocate prediction output
    submission_format = pd.read_csv(DATA_PATH / "submission_format.csv", index_col=0)
    num_labels = submission_format.shape[1]
    predict_output = np.zeros((len(test_dataset), num_labels))

    # Perform (and time) inference
    inference_start = datetime.now()
    t0 = datetime.now()
    losses = AverageMeter()
    criterion = MultiLabelSoftMarginLoss()
    for idx, batch in enumerate(test_dataloader):
        if idx % 20 == 0 and idx != 0:
            logging.info(f"batch {idx}, {datetime.now() - t0}")
            logging.info(f"{losses.avg:0.5f}")
            t0 = datetime.now()

        outputs = []
        # for model in models:

        for i in range(2):
            imgs = batch[f"images{i+1}"][0]  # .type(torch.FloatTensor).cuda()
            mirror = torch.flip(imgs, (3,))
            imgs_mirror = torch.cat([imgs, mirror], dim=0).type(torch.FloatTensor).cuda()

            output = torch.sigmoid(models[i](imgs_mirror))
            arr = output.cpu().numpy()
            model_predict = gmean(arr, axis=0)
            outputs.append(model_predict)

        mean_arr = np.array(outputs)
        preds = gmean(mean_arr, axis=0)

        # targets = batch["label"].cuda()
        # output = torch.from_numpy(preds).cuda().unsqueeze(0)
        # logits = torch.log(output / (1 - output + 1e-7))
        # loss = criterion(logits, targets)
        #
        # reduced_loss = loss.data
        # losses.update(to_python_float(reduced_loss), 1)

        predict_output[BATCH_SIZE * idx : BATCH_SIZE * (idx + 1)] = preds

    # logging.info(f"final {losses.avg:0.5f}")

    inference_stop = datetime.now()
    logging.info(f"Inference complete. Took {inference_stop - inference_start}.")
    logging.info("Creating submission.")

    # Check our predictions are in the same order as the submission format
    # assert np.all(
    #     test_metadata.seq_id.unique().tolist() == submission_format.index.to_list()
    # )

    my_submission = pd.DataFrame(
        predict_output,
        # Remember that we are predicting at the sequence, not image level
        index=test_dataset.test_metadata.seq_id,
        columns=submission_format.columns,
    )

    # We want to ensure all of our data are floats, not integers
    my_submission = my_submission.astype(np.float)

    # Save out submission to root of directory
    my_submission.to_csv("submission.csv", index=True)
    logging.info(f"Submission saved.")


if __name__ == "__main__":
    perform_inference()
