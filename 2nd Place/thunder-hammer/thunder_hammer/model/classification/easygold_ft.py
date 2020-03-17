import os
from easygold.models.senet import pd_senet103
from easygold.models.resnet import resnext101_32x16d, resnext101_32x8d
from pretrainedmodels.models import nasnetalarge
import torch.nn as nn
import torch
from torch.autograd import Variable


class PDSENet102_FT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = pd_senet103(pretrained=True)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        new_last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
        new_last_linear.weight.data = self.model.last_linear.weight.data[:num_classes]
        new_last_linear.bias.data = self.model.last_linear.bias.data[:num_classes]
        self.model.last_linear = new_last_linear

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self, num_stages=0):
        if num_stages >= 1:
            for m in [self.model.layer0]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, num_stages + 1):
            m = getattr(self.model, "layer{}".format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


class resnext101_32x8d_ft(nn.Module):
    def __init__(self, num_classes, weights=False):
        super().__init__()
        self.model = resnext101_32x8d(pretrained=True)
        # self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        new_fc = nn.Linear(self.model.fc.in_features, num_classes)
        new_fc.weight.data = self.model.fc.weight.data[:num_classes]
        new_fc.bias.data = self.model.fc.bias.data[:num_classes]
        self.model.fc = new_fc

        if weights:
            state_dict = torch.load(weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                sanitized = {}
                for k, v in state_dict.items():
                    sanitized[k.replace("model.", "").replace("last_linear", "fc")] = v

                state_dict = sanitized

            self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self, num_stages=0):
        if num_stages >= 1:
            for m in [self.model.conv1, self.model.bn1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, num_stages + 1):
            m = getattr(self.model, "layer{}".format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


class nasnetalarge_ft(nn.Module):
    def __init__(self, num_classes, weights=False):
        super().__init__()
        self.model = nasnetalarge(num_classes=1000, pretrained="imagenet")
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        new_fc = nn.Linear(self.model.last_linear.in_features, num_classes)
        new_fc.weight.data = self.model.last_linear.weight.data[:num_classes]
        new_fc.bias.data = self.model.last_linear.bias.data[:num_classes]
        self.model.last_linear = new_fc

        if weights:
            state_dict = torch.load(weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                sanitized = {}
                for k, v in state_dict.items():
                    sanitized[k.replace("model.", "").replace("last_linear", "fc")] = v

                state_dict = sanitized

            self.model.load_state_dict(state_dict, strict=True)

        # self.model = torch.jit.load('/media/n01z3/ssd1_intel/code/drivendata-identify-wildlife/src/tmp/nna.pth')

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self, num_stages=0):
        if num_stages >= 1:
            for m in [self.model.conv0, self.model.bn1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, num_stages + 1):
            m = getattr(self.model, "layer{}".format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    model = nasnetalarge_ft(54)
    sample = Variable(torch.randn(16, 3, 192, 256))

    output = model(sample)
    print(output.size())

    scripted_model = torch.jit.trace(model, sample)

    os.makedirs("assets", exist_ok=True)
    scripted_model.save("/media/n01z3/ssd1_intel/code/drivendata-identify-wildlife/src/tmp/nna.pth")
