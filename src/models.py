"""Arquitecturas CNN para transfer learning.

1. set_parameter_requires_grad()
2. build_pothole_inception()
3. build_pothole_resnet18()
4. unfreeze_last_layers()
5. count_trainable_parameters()

"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def build_pothole_inception(num_classes: int = 2, dropout: float = 0.4, feature_extract: bool = True):
    """Crea InceptionV3 preentrenado en ImageNet para clasificación binaria/multiclase.

    Durante entrenamiento InceptionV3 puede devolver logits principales y auxiliares.
    El código de entrenamiento maneja ambos casos.
    """
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, aux_logits=True)

    set_parameter_requires_grad(model, feature_extract)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    if model.AuxLogits is not None:
        aux_in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    return model


def build_pothole_resnet18(num_classes: int = 2, dropout: float = 0.4, feature_extract: bool = True):
    """Alternativa más liviana y rápida que InceptionV3."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    set_parameter_requires_grad(model, feature_extract)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_last_layers(model, blocks: list = None):
    """Descongela bloques lógicos de InceptionV3 por nombre.

    blocks: lista de nombres de módulos hijo, e.g. ["Mixed_7b", "Mixed_7c"].
    Si blocks es None solo se desbloquea el clasificador fc.
    """
    if blocks is None:
        blocks = []

    for name, module in model.named_children():
        if name in blocks:
            for param in module.parameters():
                param.requires_grad = True
            print(f"  Descongelado: {name}")

    for param in model.fc.parameters():
        param.requires_grad = True

    if getattr(model, "AuxLogits", None) is not None:
        for param in model.AuxLogits.fc.parameters():
            param.requires_grad = True

    return model


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
