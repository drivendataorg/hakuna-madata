"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import os.path as osp
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from addict import Dict
from fire import Fire
from thunder_hammer.utils import fit, set_determenistic, object_from_dict


class ClassificationPipeline(pl.LightningModule):
    def __init__(self, hparams):
        super(ClassificationPipeline, self).__init__()

        self.hparams = hparams
        self.model = object_from_dict(self.hparams.model)
        self.criterion = object_from_dict(self.hparams.loss)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            acc1 = acc1.unsqueeze(0)
            acc5 = acc5.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "acc1": acc1, "acc5": acc5, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output


    # def training_end(self, *args, **kwargs):


    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.forward(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            acc1 = acc1.unsqueeze(0)
            acc5 = acc5.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val, "val_acc1": acc1, "val_acc5": acc5})

        return output

    def validation_end(self, outputs):

        tqdm_dict = {}

        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": tqdm_dict["val_loss"]}
        return result

    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.optimizer.lr,
        #     momentum=self.hparams.optimizer.momentum,
        #     weight_decay=self.hparams.optimizer.weight_decay,
        # )

        optimizer = object_from_dict(self.hparams.optimizer, params=self.parameters())

        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return object_from_dict(self.hparams.train_data, mode="train")

    @pl.data_loader
    def val_dataloader(self):
        return object_from_dict(self.hparams.val_data, mode="val")


def main(hparams):
    if hparams.seed:
        set_determenistic(hparams.seed)

    dump_folder = osp.join(hparams.dump_path, f"{hparams.name}_{hparams.version}")
    weights_path = osp.join(dump_folder, f"weights_{0}")
    log_path = osp.join(dump_folder, f"logs")

    pipeline = ImageNetLightningModel(hparams)

    trainer = object_from_dict(
        hparams.trainer,
        checkpoint_callback=object_from_dict(hparams.checkpoint, filepath=weights_path),
        logger=object_from_dict(hparams.logger, path=log_path, run_name=f"base_{0}", version=hparams.version),
    )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(pipeline)


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    main(cfg)
