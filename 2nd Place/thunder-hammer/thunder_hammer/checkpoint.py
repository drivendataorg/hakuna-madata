from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import os.path as osp
import warnings
import logging


class JustModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_top_k=1,
        save_weights_only=False,
        mode="auto",
        period=1,
        prefix="",
    ) -> None:
        super(JustModelCheckpoint, self).__init__(
            filepath, monitor, verbose, save_top_k, save_weights_only, mode, period, prefix
        )
        self.best_model = ""

    def _save_link(self):
        for link in sorted(glob(osp.join(self.filepath, "*_best.pth"))):
            os.remove(link)
        link_path = osp.join(self.filepath, f"{self.best:0.5f}_best.pth")
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(osp.basename(self.best_model), link_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_check += 1

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f"{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}.ckpt"
            version_cnt = 0
            while os.path.isfile(filepath):
                # this epoch called before
                filepath = f"{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}_v{version_cnt}.ckpt"
                version_cnt += 1

            if self.save_top_k != -1:
                current = logs.get(self.monitor)

                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} available," " skipping.", RuntimeWarning
                    )
                else:
                    if self.check_monitor_top_k(current):

                        # remove kth
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            delpath = self.kth_best_model
                            self.best_k_models.pop(self.kth_best_model)
                            self._del_model(delpath)

                        self.best_k_models[filepath] = current
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            # monitor dict has reached k elements
                            if self.mode == "min":
                                self.kth_best_model = max(self.best_k_models, key=self.best_k_models.get)
                            else:
                                self.kth_best_model = min(self.best_k_models, key=self.best_k_models.get)
                            self.kth_value = self.best_k_models[self.kth_best_model]

                        if self.mode == "min":
                            self.best = min(self.best_k_models.values())
                            self.best_model = min(self.best_k_models, key=self.best_k_models.get)
                        else:
                            self.best = max(self.best_k_models.values())
                            self.best_model = max(self.best_k_models, key=self.best_k_models.get)

                        if self.verbose > 0:
                            logging.info(
                                f"\nEpoch {epoch:05d}: {self.monitor} reached"
                                f" {current:0.5f} (best {self.best:0.5f}), saving model to"
                                f" {filepath} as top {self.save_top_k}"
                            )
                        self._save_model(filepath)
                        self._save_link()

                    else:
                        if self.verbose > 0:
                            logging.info(
                                f"\nEpoch {epoch:05d}: {self.monitor}"
                                f" was not in top {self.save_top_k}, best: {self.best:0.5f}"
                            )

            else:
                if self.verbose > 0:
                    logging.info(f"\nEpoch {epoch:05d}: saving model to {filepath}")
                self._save_model(filepath)
