from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lesion_ml.models.forward import forward_batch


class ImageOnlyModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones(image.shape[0], 7, device = image.device)
    
class MetadataModel(nn.Module):
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        assert metadata.ndim == 2
        assert image.shape[0] == metadata.shape[0]
        return torch.ones(image.shape[0], 7, device = image.device)
    
def test_forward_batch_image_only() -> None:
    device = torch.device("cpu")        
    model  = ImageOnlyModel()

    batch: dict[str, Any] = {
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.zeros(4, dtype=torch.long),
    }

    logits = forward_batch(model, batch, device)

    assert logits.shape == (4, 7)

def test_forward_batch_with_metadata() -> None:
    device = torch.device("cpu")
    model = MetadataModel()

    batch: dict[str, Any] = {
        "image": torch.randn(4, 3, 224, 224),
        "metadata": torch.randn(4, 10),
        "label": torch.zeros(4, dtype=torch.long),
    }

    logits = forward_batch(model, batch, device)

    assert logits.shape == (4, 7)


