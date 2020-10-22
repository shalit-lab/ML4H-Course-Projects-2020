from typing import Any

import torch
from torch import nn
import torchvision.models as models

from functions import ReverseGradient


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18()
        modules = list(resnet.children())[:6]
        self.feature_extractor = nn.Sequential(*modules)

    def forward(self, x):
        # output feature map with 128 channels
        return self.feature_extractor(x)
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class Classifier(nn.Module):
    def __init__(self, output_dim, apply_reverse_gradient=False):
        super(Classifier, self).__init__()
        self.apply_reverse_gradient = apply_reverse_gradient
        resnet = models.resnet18()
        childrens =  list(resnet.children())
        modules = childrens[6:len(childrens)-1] + [
            Flatten(),
            resnet.fc,
            nn.Linear(1000, output_dim)
        ]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.apply_reverse_gradient:
            x = ReverseGradient.apply(x)
        return self.model(x)


class SingleBiasDetectorModel(nn.Module):
    def __init__(self):
        super(SingleBiasDetectorModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.primary_classifier = Classifier(output_dim=3)
        self.bias_classifier = Classifier(output_dim=1, apply_reverse_gradient=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.primary_classifier(x)
        bias = self.bias_classifier(x).squeeze()
        return output, bias

class MultipleBiasDetectorModel(nn.Module):
    def __init__(self):
        super(MultipleBiasDetectorModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.primary_classifier = Classifier(output_dim=3)
        self.red_line_bias_classifier = Classifier(output_dim=1, apply_reverse_gradient=True)
        self.blue_polygon_bias_classifier = Classifier(output_dim=1, apply_reverse_gradient=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        output = self.primary_classifier(x)
        red_line_bias = self.red_line_bias_classifier(x).squeeze()
        blue_polygon_bias = self.blue_polygon_bias_classifier(x).squeeze()
        return output, red_line_bias, blue_polygon_bias
