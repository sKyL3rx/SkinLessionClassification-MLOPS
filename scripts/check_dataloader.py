import yaml
from torch.utils.data import DataLoader

from src.lesion_ml.data.dataset import SkinLesionDataset, build_label_mapping_from_csv
from src.lesion_ml.data.transforms import build_transforms_from_config

with open("params.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)


label_to_idx = build_label_mapping_from_csv("data/splits/train.csv")

train_ds = SkinLesionDataset(
    csv_path="data/splits/train.csv",
    transform=build_transforms_from_config(config, "train"),
    label_to_idx=label_to_idx,
)

loader = DataLoader(train_ds, batch_size=4, shuffle=True)
batch = next(iter(loader))

print(batch["image"].shape)
print(batch["label"].shape)
print(batch["label_name"][:4])
