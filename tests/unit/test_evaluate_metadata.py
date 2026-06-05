from __future__ import annotations

from typing import Any

import torch
from torch import nn

from scripts.evaluate import apply_tta_transform, get_tta_config, predict_proba


class MetadataEvalModel(nn.Module):
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(image.shape[0], 3, device=image.device)
        logits[:, 1] = metadata[:, 0]
        return logits

def test_get_tta_config_disables_transforms_when_use_tta_false() -> None:
    config: dict[str,Any] = {
        "evaluate": {
            "use_tta": False,
            "tta_transforms": ["original", "hflip", "vflip"],
        }
    }

    use_tta, transforms = get_tta_config(config)

    assert use_tta is False
    assert transforms == ["original"]

def test_apply_tta_hflip() -> None:
    images = torch.arange(1*1*2*3).reshape(1,1,2,3)

    flipped = apply_tta_transform(images, "hflip")

    assert torch.equal(flipped, torch.flip(images, dims=[3]))

def test_predict_proba_metadata_with_tta() -> None:
    model = MetadataEvalModel()
    device = torch.device("cpu")

    batch: dict[str, Any] = {
        "image": torch.randn(4, 3, 32, 32),
        "metadata": torch.ones(4, 5),
        "label": torch.zeros(4, dtype=torch.long),
    }
    config: dict[str, Any] = {
        "evaluate": {
            "use_tta": True,
            "tta_transforms": ["original", "hflip", "vflip"],
        }
    }

    probs = predict_proba(
        model=model,
        batch=batch,
        device=device,
        config=config,
    )

    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4))