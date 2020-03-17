import os
import torch
import numpy as np
import os.path as osp
from glob import glob
from thunder_hammer.utils import fit, set_determenistic, update_config, object_from_dict
import yaml


def main():
    cfg_path = "../../configs/rx50_stages_7.yml"
    batch_size = 1

    with open(cfg_path) as file:
        cfg = yaml.full_load(file)

    print(cfg["model"])
    model = object_from_dict(cfg["model"])

    checkpoint_path = "/mnt/hdd1/learning_dumps/wild/rx50_stages_7/weights_stage4/_ckpt_epoch_6.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    sanitized_dict = {}
    for k, v in checkpoint["state_dict"].items():
        # sanitized_dict[k.replace("model.", "")] = v
        # sanitized_dict[k.replace("model.model", "model")] = v
        if "101" in cfg_path:
            sanitized_dict[k.replace("model.model", "model")] = v
        else:
            # sanitized[k.replace("model.model", "model")] =
            sanitized_dict[k.replace("model.", "")] = v

    model.load_state_dict(sanitized_dict)

    sample = torch.rand(batch_size, 3, 360, 480, dtype=torch.float32)
    print(sample.shape)

    scripted_model = torch.jit.trace(model, sample)

    os.makedirs("assets", exist_ok=True)
    scripted_model.save("assets/rx50_v7_s4_e6.pth")


if __name__ == "__main__":
    main()
