"""Carga de datos, EDA y DataLoaders para el proyecto de detección de huecos.

Dataset principal: taroii/pothole-detection (Hugging Face).
El problema se formula como clasificación binaria: pothole vs no pothole.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetInfo:
    dataset: DatasetDict
    image_key: str
    label_key: str
    class_names: List[str]
    num_classes: int


class PotholeDataset(Dataset):
    """Wrapper de un split de Hugging Face para usarlo con PyTorch."""

    def __init__(self, hf_dataset, image_key: str, label_key: str, class_names: List[str], transform=None):
        self.data = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.class_names = class_names
        self.transform = transform

        first_label = hf_dataset[0][label_key]
        self._string_labels = isinstance(first_label, str)
        self._label_map = {name: i for i, name in enumerate(class_names)} if self._string_labels else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = item[self.image_key]

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        label = item[self.label_key]
        if self._string_labels:
            label = self._label_map[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)


def load_pothole_dataset(dataset_name: str = "taroii/pothole-detection") -> DatasetInfo:
    """Carga el dataset y detecta automáticamente columnas de imagen/etiqueta."""
    ds = load_dataset(dataset_name)
    first_split = list(ds.keys())[0]
    sample = ds[first_split][0]

    image_key = None
    for key, value in sample.items():
        if isinstance(value, Image.Image):
            image_key = key
            break
    if image_key is None:
        raise ValueError("No se pudo detectar una columna de imagen PIL en el dataset.")

    label_candidates = ["label", "labels", "class", "category"]
    label_key = next((k for k in label_candidates if k in sample.keys()), None)
    if label_key is None:
        label_key = next(k for k in sample.keys() if k != image_key)

    feature = ds[first_split].features[label_key]
    if hasattr(feature, "names") and feature.names is not None:
        class_names = list(feature.names)
    else:
        labels = ds[first_split][label_key]
        unique_labels = sorted(set(labels))
        if isinstance(unique_labels[0], str):
            class_names = unique_labels
        else:
            class_names = [str(i) for i in unique_labels]

    return DatasetInfo(
        dataset=ds,
        image_key=image_key,
        label_key=label_key,
        class_names=class_names,
        num_classes=len(class_names),
    )


def get_transforms(image_size: int = 299, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Transforms para entrenamiento y evaluación.

    InceptionV3 usa 299x299. Para ResNet/EfficientNet pueden usar 224.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, eval_tfms


def create_dataloaders(
    dataset_name: str = "taroii/pothole-detection",
    image_size: int = 299,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool | None = None,
):
    info = load_pothole_dataset(dataset_name)
    ds = info.dataset

    required = {"train", "validation", "test"}
    missing = required - set(ds.keys())
    if missing:
        raise ValueError(f"El dataset no trae estos splits requeridos: {missing}")

    train_tfms, eval_tfms = get_transforms(image_size)

    train_ds = PotholeDataset(ds["train"], info.image_key, info.label_key, info.class_names, train_tfms)
    val_ds = PotholeDataset(ds["validation"], info.image_key, info.label_key, info.class_names, eval_tfms)
    test_ds = PotholeDataset(ds["test"], info.image_key, info.label_key, info.class_names, eval_tfms)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, info


def class_distribution(info: DatasetInfo) -> Dict[str, Counter]:
    out = {}
    for split in info.dataset.keys():
        labels = info.dataset[split][info.label_key]
        if len(labels) == 0:
            out[split] = Counter()
            continue
        if isinstance(labels[0], str):
            counts = Counter(labels)
        else:
            counts = Counter(info.class_names[int(x)] for x in labels)
        out[split] = counts
    return out


def plot_class_distribution(info: DatasetInfo, save_path: str | None = None):
    counts_by_split = class_distribution(info)
    splits = list(counts_by_split.keys())

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        counts = counts_by_split[split]
        classes = list(counts.keys())
        values = list(counts.values())
        bars = ax.bar(classes, values)
        ax.set_title(f"Split: {split}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Cantidad de imágenes")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha="center", va="bottom")

    plt.suptitle("Distribución de clases — Pothole Detection Dataset", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()


def show_samples_by_class(info: DatasetInfo, split: str = "train", samples_per_class: int = 4, save_path: str | None = None):
    split_data = info.dataset[split]
    indices_by_class = {cls: [] for cls in info.class_names}

    for idx, label in enumerate(split_data[info.label_key]):
        cls_name = label if isinstance(label, str) else info.class_names[int(label)]
        indices_by_class[cls_name].append(idx)

    fig, axes = plt.subplots(info.num_classes, samples_per_class, figsize=(3 * samples_per_class, 3 * info.num_classes))
    if info.num_classes == 1:
        axes = np.array([axes])
    if samples_per_class == 1:
        axes = axes.reshape(info.num_classes, 1)

    for row, cls in enumerate(info.class_names):
        available = indices_by_class[cls]
        chosen = random.sample(available, min(samples_per_class, len(available)))
        for col in range(samples_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(chosen):
                img = split_data[chosen[col]][info.image_key]
                ax.imshow(img)
            if col == 0:
                ax.set_title(cls, fontweight="bold")

    plt.suptitle("Muestras del dataset por clase", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()


def image_dimension_report(info: DatasetInfo, split: str = "train", n_samples: int = 100, save_path: str | None = None) -> pd.DataFrame:
    split_data = info.dataset[split]
    idxs = random.sample(range(len(split_data)), min(n_samples, len(split_data)))

    rows = []
    for idx in idxs:
        image = split_data[idx][info.image_key]
        arr = np.array(image)
        rows.append({
            "width": image.size[0],
            "height": image.size[1],
            "channels": arr.shape[2] if arr.ndim == 3 else 1,
        })

    df = pd.DataFrame(rows)
    print(df.describe())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["height"], bins=20)
    axes[0].set_title("Distribución de alturas")
    axes[0].set_xlabel("Píxeles")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(df["width"], bins=20)
    axes[1].set_title("Distribución de anchuras")
    axes[1].set_xlabel("Píxeles")
    axes[1].set_ylabel("Frecuencia")

    plt.suptitle("Distribución de dimensiones de imagen")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()

    return df


def pixel_statistics_report(info: DatasetInfo, split: str = "train", n_samples: int = 200, save_path: str | None = None):
    split_data = info.dataset[split]
    idxs = random.sample(range(len(split_data)), min(n_samples, len(split_data)))

    pixels = {0: [], 1: [], 2: []}
    for idx in idxs:
        img = split_data[idx][info.image_key].convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        for c in range(3):
            pixels[c].append(arr[:, :, c].reshape(-1))

    stats = {}
    names = ["R", "G", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for c, ax in enumerate(axes):
        values = np.concatenate(pixels[c])
        stats[names[c]] = {"mean": float(values.mean()), "std": float(values.std())}
        ax.hist(values, bins=50)
        ax.axvline(values.mean(), linestyle="--", label=f"media={values.mean():.3f}")
        ax.set_title(f"Canal {names[c]}")
        ax.set_xlabel("Intensidad normalizada")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=9)

    plt.suptitle("Distribución de intensidad por canal de color")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.show()

    print("Estadísticas RGB:")
    for k, v in stats.items():
        print(f"Canal {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
    return stats
