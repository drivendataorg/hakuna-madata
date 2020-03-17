import logging
import os
import warnings

import apex
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from addict import Dict
from fire import Fire

from thunder_hammer.utils import fit, set_determenistic, object_from_dict, reduce_tensor

# warnings.filterwarnings("ignore")
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"


class ImageNetLightningPipeline(pl.LightningModule):
    def __init__(self, hparams, model=None, optimizers=None, verbose=False):
        super(ImageNetLightningPipeline, self).__init__()

        self.hparams = hparams
        self.model = model
        if model is None:
            logging.info("model created from config")
            self.model = object_from_dict(self.hparams.model)
        if hparams.sync_bn:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.loss = object_from_dict(self.hparams.loss)
        self.optimizers = optimizers
        self.scheduler = None
        self.every_step = True
        self.len_epoch = 0
        self.forward_pass = object_from_dict(self.hparams.mixup, criterion=self.loss)
        self.train_loader = None
        self.val_loader = None
        self.metrics = [object_from_dict(el) for el in self.hparams.metrics]
        self.verbose = verbose

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        if self.every_step:
            self.scheduler.step(self.current_epoch, self.global_step, self.len_epoch)

        images, targets = batch
        outputs = self.forward(images)
        loss = self.loss(outputs, targets)
        loss_data = reduce_tensor(loss, self.hparams.trainer.gpus)
        # loss_val, outputs = self.forward_pass.step(self.model, images, targets)
        metrics = self._get_metrics_dict(outputs, targets, mode="train")
        metrics["lr"] = self._get_current_lr()
        return {"loss": loss_data, "progress_bar": metrics, "log": metrics}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metrics = self._get_metrics_dict(outputs, targets, mode="val")
        return metrics

    def validation_end(self, outputs):
        mean_metrics = self._get_mean_metrics(outputs)
        self._purge_loader()
        return {"val_loss": mean_metrics["val_loss"], "progress_bar": mean_metrics, "log": mean_metrics}

    def _get_metrics_dict(self, logits: torch.Tensor, labels: torch.Tensor, mode: str):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Got unknown mode: {mode}")
        result = {}
        for metric in self.metrics:
            if metric.name == "loss":
                raise ValueError("Naming metric as loss is forbidden")
            metric_tensor = metric(logits, labels)
            result[f"{mode}_{metric.name}"] = reduce_tensor(metric_tensor, self.hparams.trainer.gpus)
        # result[f"{mode}_loss"] = self.loss(logits, labels)
        return result

    def _get_mean_metrics(self, metrics_list):
        if len(metrics_list) == 0:
            raise ValueError("Got empty metrics_list to average")
        metric_names = metrics_list[0].keys()

        mean_metrics = {}
        for metric_name in metric_names:
            mean_metrics[metric_name] = torch.stack([metrics[metric_name] for metrics in metrics_list]).mean()
        return mean_metrics

    def configure_optimizers(self):
        parameters = list(self.model.named_parameters())
        if self.hparams.no_bn_weight_decay:
            bn_params = [v for n, v in parameters if "bn" in n]
            rest_params = [v for n, v in parameters if not "bn" in n]
            if torch.distributed.get_rank() == 0:
                logging.info(f"no_bn_wd, bn:{len(bn_params)} rest{len(rest_params)}")

            params = [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": self.hparams.optimizer.weight_decay},
            ]
        else:
            params = filter(lambda x: x.requires_grad, self.model.parameters())

        if self.optimizers is None:
            self.optimizers = [object_from_dict(self.hparams.optimizer, params=params)]

        if self.hparams.freeze:
            self.model.freeze(self.hparams.freeze)

        self.scheduler = object_from_dict(self.hparams.scheduler, optimizer=self.optimizers[0])
        if "CosineWarmRestart" in self.hparams.scheduler.type:
            self.every_step = True
            return self.optimizers
        else:
            self.every_step = False
            return self.optimizers, [self.scheduler]

    @pl.data_loader
    def train_dataloader(self):
        self.train_loader = object_from_dict(self.hparams.train_data, mode="train")
        self.len_epoch = len(self.train_loader)
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        self.val_loader = object_from_dict(self.hparams.val_data, mode="val")
        return self.val_loader

    def load_weights_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])

    def configure_apex(self, amp, model, optimizers, amp_level):
        model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level, loss_scale=self.hparams.loss_scale)
        return model, optimizers

    def _get_current_lr(self):
        return list(map(lambda group: group["lr"], self.optimizers[0].param_groups))[0]

    def _purge_loader(self):
        if self.hparams.train_data.type.split(".")[-1] == "DaliLoader":
            if self.verbose:
                logging.info("purge train loader")
            for batch in self.train_loader:
                pass
        if self.hparams.val_data.type.split(".")[-1] == "DaliLoader":
            if self.verbose:
                logging.info("purge val loader")
            for batch in self.val_loader:
                pass

    def _freeze_encoder(self) -> None:
        found_encoder = False
        for encoder_name in ("encoder_stages", "encoder", "encoder_features", "features"):
            if hasattr(self._model, encoder_name):
                found_encoder = True
                for p in getattr(self._model, encoder_name).parameters():
                    p.requires_grad = False
        if not found_encoder:
            raise AttributeError("No encoder or encoder_stages, idk what to freeze")

    @classmethod
    def load_from_checkpoint_params(cls, checkpoint_path, hparams):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model = cls(hparams)
        model.load_state_dict(checkpoint["state_dict"])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model


def main(hparams):
    if hparams.seed:
        set_determenistic(hparams.seed)

    pipeline = ImageNetLightningPipeline(hparams)

    trainer = object_from_dict(
        hparams.trainer,
        checkpoint_callback=object_from_dict(hparams.checkpoint),
        logger=object_from_dict(hparams.logger),
    )

    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(pipeline)


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    main(cfg)
