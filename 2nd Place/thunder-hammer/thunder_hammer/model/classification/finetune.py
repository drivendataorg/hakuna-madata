import torch.nn as nn
from pretrainedmodels import nasnetalarge, senet154, resnet18
import torch
from torch.autograd import Variable


class NASNetALargeFT(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.model = nasnetalarge(pretrained=pretrained, num_classes=1000)
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
            for m in [self.model.conv0]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        if num_stages >= 2:
            for m in [self.model.cell_stem_0, self.model.cell_stem_1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


class SENet154FT(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
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
            for m in [self.model.layer0_modules]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


def get_se154(num_classes):
    net = SENet154FT(num_classes)
    return net


if __name__ == "__main__":
    model = SENet154FT(num_classes=54).to(torch.device('cuda:1'))
    input = Variable(torch.randn(64, 3, 256, 256)).to(torch.device('cuda:1'))

    while True:
        output = model(input)
        print(output.size())
