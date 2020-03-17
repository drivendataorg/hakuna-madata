import os
import os.path as osp
from distutils.dir_util import copy_tree
import json
import pandas as pd
import yaml
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only


class JustLogger(LightningLoggerBase):
    def __init__(self, path: str, run_name: str, version: int) -> None:
        super(JustLogger, self).__init__()
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.run_name = run_name
        self.df = pd.DataFrame()
        self.yaml_path = osp.join(self.path, f"{self.run_name}.yml")
        self.json_path = osp.join(self.path, f"{self.run_name}.json")
        self.df_path = osp.join(self.path, f"{self.run_name}.csv")
        self._version = version

    @rank_zero_only
    def log_hyperparams(self, params):
        with open(self.yaml_path, "w") as outfile:
            yaml.dump(params, outfile, default_flow_style=True)

        with open(self.json_path, "w") as fp:
            json.dump(params, fp)

    @rank_zero_only
    def save(self):
        dst = osp.join(self.path, "code")
        os.makedirs(dst, exist_ok=True)
        copy_tree("./", dst)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.rank > 0 or len(metrics) < 2:
            return

        tdf = pd.DataFrame()
        tdf["step"] = [step]
        for k, v in metrics.items():
            tdf[k] = [v]

        self.df = pd.concat([self.df, tdf])
        self.df.to_csv(self.df_path, index=False)

    @property
    def version(self):
        return self._version  # FIXME
