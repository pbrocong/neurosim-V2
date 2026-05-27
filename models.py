# models.py
#
# All weights are *bounded* in [weight_min, weight_max] (typically [-1, 1])
# because they map to a single conductance cell in the crossbar.
#
# The previous version used `nn.init.uniform_(w, weight_min, weight_max)`,
# which for a 5x5x1 conv kernel produces an activation std ≈ √(25/3) ≈ 2.89
# and for a 784-wide MLP weights produces activation std ≈ √(784/3) ≈ 16.
# That saturates tanh/ReLU on the first forward pass and kills CNN training
# completely.
#
# Fix: keep the weights inside the bounded range but use **Glorot-uniform
# inside that range**, scaled by 1/√fan_in (or 1/√((fan_in+fan_out)/2) for
# Linear). The init bound is `clip(scaled_bound, 0, (weight_max-weight_min)/2)`
# so we never violate the hardware range.

import math
import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Init helper
# ---------------------------------------------------------------------------
def _bounded_glorot_(weight: torch.Tensor, w_min: float, w_max: float,
                     gain: float = 1.0):
    """Glorot-uniform init, clipped to [w_min, w_max].

    Computes the canonical Glorot bound and shrinks it (never grows) so we
    stay inside the hardware weight range.
    """
    if weight.dim() < 2:
        bound = 1.0 / math.sqrt(weight.shape[0]) * gain
    else:
        fan_in = weight.shape[1] * (weight[0][0].numel() if weight.dim() > 2 else 1)
        fan_out = weight.shape[0] * (weight[0][0].numel() if weight.dim() > 2 else 1)
        bound = gain * math.sqrt(6.0 / max(fan_in + fan_out, 1))
    hw_half = 0.5 * (w_max - w_min)
    centre = 0.5 * (w_max + w_min)
    use_bound = min(bound, hw_half)
    with torch.no_grad():
        weight.uniform_(centre - use_bound, centre + use_bound)


def initialize_weights_uniform(model: nn.Module, weight_min: float, weight_max: float,
                               legacy_full_range: bool = False):
    """Init every Conv2d/Linear in `model`.

    Set legacy_full_range=True to reproduce the *old* broken behaviour
    (uniform over the entire [w_min, w_max] range) for reference runs.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if legacy_full_range:
                nn.init.uniform_(m.weight, weight_min, weight_max)
            else:
                _bounded_glorot_(m.weight, weight_min, weight_max)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 1) MNIST-friendly MLP  (log_softmax output, NLLLoss)
# ---------------------------------------------------------------------------
class SimpleNet(nn.Module):
    """Small MLP, V3.0-faithful (matches MLP_NeuroSim's 400/100/10-style FC stack
    in spirit). We keep 28x28 -> 128 -> num_classes here for parity with the
    existing pipeline. BN can be toggled via use_bn for ablation."""

    def __init__(self, weight_min: float, weight_max: float,
                 use_bn: bool = False, num_classes: int = 10):
        super().__init__()
        self.use_bn = bool(use_bn)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128) if self.use_bn else nn.Identity()
        self.fc2 = nn.Linear(128, num_classes)

        _bounded_glorot_(self.fc1.weight, weight_min, weight_max)
        _bounded_glorot_(self.fc2.weight, weight_min, weight_max)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# 2) CNN family  (logits output, CrossEntropyLoss)
# ---------------------------------------------------------------------------
class Simple_CNN(nn.Module):
    """1 conv layer + 1 FC. Default for ASL."""

    def __init__(self, weight_min: float, weight_max: float, num_classes: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        initialize_weights_uniform(self, weight_min, weight_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
        return x


class Standard_CNN(nn.Module):
    """2 conv layers + small MLP head."""

    def __init__(self, weight_min: float, weight_max: float, num_classes: int = 24):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
        initialize_weights_uniform(self, weight_min, weight_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ResNet18_MNIST(weight_min: float, weight_max: float, num_classes: int = 24):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    initialize_weights_uniform(model, weight_min, weight_max)
    return model


class VGG_MNIST(nn.Module):
    def __init__(self, weight_min: float, weight_max: float, num_classes: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        initialize_weights_uniform(self, weight_min, weight_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class AlexNet_MNIST(nn.Module):
    def __init__(self, weight_min: float, weight_max: float, num_classes: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        initialize_weights_uniform(self, weight_min, weight_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class LeNet5_MNIST(nn.Module):
    def __init__(self, weight_min: float, weight_max: float, num_classes: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        initialize_weights_uniform(self, weight_min, weight_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
