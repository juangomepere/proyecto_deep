"""Grad-CAM simple para explicar predicciones del modelo."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def _get_module_by_name(model, layer_name: str):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = list(modules.keys())[-30:]
        raise ValueError(f"No existe la capa '{layer_name}'. Algunas capas disponibles al final: {available}")
    return modules[layer_name]


def attach_hooks(model, layer_name: str):
    """Adjunta hooks a una capa y retorna diccionario de capturas y función remove()."""
    layer = _get_module_by_name(model, layer_name)
    capture: Dict[str, torch.Tensor] = {}

    def forward_hook(module, inputs, output):
        capture["act"] = output

    def backward_hook(module, grad_input, grad_output):
        capture["grad"] = grad_output[0]

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    def remove():
        h1.remove()
        h2.remove()

    return capture, remove


def run_forward(model, input_tensor: torch.Tensor):
    output = model(input_tensor)
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    pred = int(logits.argmax(dim=1).item())
    return logits, pred


def run_backward(output: torch.Tensor, class_idx: int):
    score = output[:, class_idx].sum()
    score.backward(retain_graph=True)


def compute_gradcam(activation: torch.Tensor, gradient: torch.Tensor):
    """Calcula Grad-CAM a partir de activaciones y gradientes capturados."""
    # activation/gradient: [B, C, H, W]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def resize_cam(cam, height: int, width: int):
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    return torch.tensor(cv2.resize(cam, (width, height)), dtype=torch.float32)


def tensor_to_uint8(image_tensor: torch.Tensor):
    """Convierte tensor [C,H,W] en [0,1] a uint8 [H,W,C]."""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_cam(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    cam = np.clip(cam, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * image_uint8 + alpha * heatmap)
    return overlay, heatmap
