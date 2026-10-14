import torch.nn as nn
from torchvision import models

def build_model(model_name: str, dropout: float = 0.3):
    # Get official weights enum safely
    weights_enum = models.get_model_weights(model_name).DEFAULT
    model = getattr(models, model_name)(weights=weights_enum)

    # --------------------------
    # EfficientNet family
    # --------------------------
    if model_name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    # --------------------------
    # ResNet family
    # --------------------------
    elif model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    # --------------------------
    # DenseNet family
    # --------------------------
    elif model_name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model