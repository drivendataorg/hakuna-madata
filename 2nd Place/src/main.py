from thunder_hammer.utils import fit, set_determenistic, update_config, object_from_dict
from thunder_hammer.stager import Stager
from addict import Dict
from fire import Fire


def main(hparams):
    if hparams.seed:
        set_determenistic(hparams.seed)

    stager = Stager(hparams)
    stager.run()


if __name__ == "__main__":
    cfg = Dict(Fire(fit))
    main(cfg)
