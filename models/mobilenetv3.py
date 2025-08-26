import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def build_mobilenetv3_small(num_classes=100, pretrained=True):
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    except Exception:
        model = mobilenet_v3_small(pretrained=pretrained)

    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model
