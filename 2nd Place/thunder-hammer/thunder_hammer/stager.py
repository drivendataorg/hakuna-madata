from glob import glob
import os
import os.path as osp
import warnings

from addict import Dict
from fire import Fire

from thunder_hammer.pipeline import ImageNetLightningPipeline
from thunder_hammer.utils import fit, set_determenistic, update_config, object_from_dict
import logging

# warnings.filterwarnings("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class Stager(object):
    def __init__(self, hparam):
        self.base_cfg = hparam
        self.version = hparam.version

        self.dump_folder = osp.join(hparam.dump_path, f"{hparam.name}_{hparam.version}")
        if "fold" in hparam.keys():
            self.dump_folder = os.path.join(self.dump_folder, f"fold_{hparam.fold}")

        self.log_path = osp.join(self.dump_folder, f"logs")
        os.makedirs(self.log_path, exist_ok=True)
        self.stages = [stage for stage in self.base_cfg.stages]
        self.resume_stage = False
        if "resume_stage" in hparam.keys():
            self.resume_stage = hparam.resume_stage

    def get_stage_weights_path(self, stage):
        weights_path = osp.join(self.dump_folder, f"weights_{stage}")
        os.makedirs(weights_path, exist_ok=True)
        return weights_path

    def get_best_previous_checkpoint(self, stage_number):
        if stage_number == 0:
            return None
        weights_path = self.get_stage_weights_path(self.stages[stage_number - 1])
        best_weights = sorted(glob(osp.join(weights_path, "*best.pth")))
        if len(best_weights) == 0:
            return None
        else:
            return best_weights[0]

    def run_stage(self, stage, number):
        logging.info(f"start {stage}")
        stage_cfg = update_config(self.base_cfg, Dict(self.base_cfg.stages[stage]))
        weights_path = self.get_stage_weights_path(stage)

        previous_checkpoint = self.get_best_previous_checkpoint(number)
        if previous_checkpoint:
            print(f"start from previous {previous_checkpoint}")
            pipeline = ImageNetLightningPipeline.load_from_checkpoint_params(
                checkpoint_path=previous_checkpoint, hparams=stage_cfg
            )
        else:
            pipeline = ImageNetLightningPipeline(stage_cfg)

        trainer = object_from_dict(
            stage_cfg.trainer,
            checkpoint_callback=object_from_dict(stage_cfg.checkpoint, filepath=weights_path),
            logger=object_from_dict(
                stage_cfg.logger, path=self.log_path, run_name=f"{stage}", version=self.base_cfg.version
            ),
        )

        trainer.fit(pipeline)
        del pipeline, trainer

    def run(self):
        starter = False
        for n, stage in enumerate(self.stages):
            if self.resume_stage:
                if stage == self.resume_stage:
                    starter = True
                if starter:
                    self.run_stage(stage, n)
            else:
                self.run_stage(stage, n)


def main(hparams):
    if hparams.seed:
        set_determenistic(hparams.seed)

    stager = Stager(hparams)
    stager.run()


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    main(cfg)
