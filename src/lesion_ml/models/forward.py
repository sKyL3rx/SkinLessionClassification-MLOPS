from __future__ import annotations

from typing import Any

import torch
from torch import nn


def forward_batch(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    images = batch["image"].to(device)

    if "metadata" in batch:
        metadata = batch["metadata"].to(device)
        return model(images, metadata)

    return model(images)