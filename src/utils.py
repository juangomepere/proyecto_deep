"""Funciones auxiliares de visualización."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def unnormalize(tensor, mean=MEAN, std=STD):
    """Revierte la normalización ImageNet de un tensor CxHxW y devuelve imagen HxWxC."""
    image = tensor.detach().cpu().clone()
    for channel, m, s in zip(image, mean, std):
        channel.mul_(s).add_(m)
    image = image.numpy().transpose(1, 2, 0)
    return np.clip(image, 0, 1)


def tensor_to_uint8_image(tensor, mean=MEAN, std=STD):
    img = unnormalize(tensor, mean, std)
    return (img * 255).astype(np.uint8)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="Entrenamiento", save_path=None):
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs, val_losses, marker="o", label="Validation Loss")
    ax.set_title(f"{title} — Loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if save_path:
        root, ext = os.path.splitext(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{root}_loss{ext or '.png'}", bbox_inches="tight", dpi=140)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    ax.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
    ax.set_title(f"{title} — Accuracy")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    if save_path:
        root, ext = os.path.splitext(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{root}_accuracy{ext or '.png'}", bbox_inches="tight", dpi=140)
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Matriz de confusión — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()


def show_batch(dataloader, class_names, n=8):
    images, labels = next(iter(dataloader))
    n = min(n, len(images))
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(unnormalize(images[i]))
            ax.set_title(class_names[int(labels[i])])
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataloader, device, class_names, n=8, save_path=None):
    model.eval()
    images, labels = next(iter(dataloader))
    images_device = images.to(device)

    with torch.no_grad():
        outputs = model(images_device)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu()

    n = min(n, len(images))
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(preds[i])]
            conf = float(probs[i, preds[i]])
            color = "green" if true_label == pred_label else "red"
            ax.imshow(unnormalize(images[i]))
            ax.set_title(f"Real: {true_label}\nPred: {pred_label} ({conf:.2f})", color=color, fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()
