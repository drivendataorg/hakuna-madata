import timm
import geffnet
import torch
import torch.nn as nn


class seresnext101_32x4d_ft(nn.Module):
    def __init__(self, num_classes, weights=False):
        super().__init__()
        self.model = timm.create_model("gluon_seresnext101_32x4d", pretrained=True)
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


class effnet_ft(nn.Module):
    def __init__(self, num_classes, weights=False):
        super().__init__()
        self.model = timm.create_model("nasnetalarge", pretrained=True)
        # print(self.model)
        # self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
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


if __name__ == "__main__":
    # print(timm)
    model = effnet_ft(54)
    print(model)

    sample = torch.randn(16, 3, 224, 320)

    output = model(sample)
    print(output.shape)
