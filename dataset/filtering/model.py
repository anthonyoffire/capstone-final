import torch.nn as nn
from torchvision import models, transforms


MODEL_TYPES = ("densenet121",)
LEARNING_RATES = {"densenet121": 1e-5}
CLASSIFICATION_THRESHOLD = 0.51
PRETRAINED_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier(
    model_type: str,
    pretrained: bool = False,
) -> nn.Module:
    if model_type == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 1), nn.Sigmoid())
        return model
    raise ValueError(f"Unknown model type {model_type!r}; choose one of {sorted(MODEL_TYPES)}.")


def get_image_transform(model_type: str):
    if model_type in MODEL_TYPES:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(PRETRAINED_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f"Unknown model type {model_type!r}; choose one of {sorted(MODEL_TYPES)}.")