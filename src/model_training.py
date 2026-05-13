"""Funciones de entrenamiento y evaluación para modelos PyTorch.

1. _get_logits()
2. train_one_epoch()
3. validate_one_epoch()
4. train_model()
5. evaluate()
6. predict_dataset()
7. classification_metrics()


"""

from __future__ import annotations

import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm


def _get_logits(outputs):
    """Devuelve logits principales aunque el modelo retorne tupla/objeto InceptionOutputs."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def train_one_epoch(model, dataloader, criterion, optimizer, device, l1_lambda: float = 0.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        logits = _get_logits(outputs)
        loss = criterion(logits, labels)

        # Si InceptionV3 devuelve logits auxiliares, se pueden usar para estabilizar entrenamiento.
        if hasattr(outputs, "aux_logits") and outputs.aux_logits is not None:
            aux_loss = criterion(outputs.aux_logits, labels)
            loss = loss + 0.4 * aux_loss

        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = _get_logits(outputs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, epochs, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(best_state, path)
            print(f"Mejor modelo guardado: {path}")

    model.load_state_dict(best_state)
    return train_losses, val_losses, train_accs, val_accs


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float]:
    loss, acc = validate_one_epoch(model, dataloader, criterion, device)
    return loss, acc


def predict_dataset(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting", leave=False):
            images = images.to(device)
            outputs = model(images)
            logits = _get_logits(outputs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def classification_metrics(model, dataloader, device, class_names=None) -> Dict:
    y_true, y_pred, y_prob = predict_dataset(model, dataloader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0) if class_names else classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
