import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BinaryAgeDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_fields = {"file_path", "binary_label"}
            missing_fields = required_fields - set(reader.fieldnames or ())
            if missing_fields:
                raise ValueError(
                    f"{self.csv_path} is missing required CSV columns: "
                    f"{', '.join(sorted(missing_fields))}."
                )

            for row_number, row in enumerate(reader, start=2):
                image_path_text = (row.get("file_path") or "").strip()
                label_text = (row.get("binary_label") or "").strip()
                if not image_path_text:
                    raise ValueError(
                        f"{self.csv_path}:{row_number} has an empty file_path."
                    )
                if label_text not in {"0", "1"}:
                    raise ValueError(
                        f"{self.csv_path}:{row_number} has invalid binary_label "
                        f"{label_text!r}; expected 0 or 1."
                    )

                image_path = Path(image_path_text).expanduser()
                if not image_path.is_absolute():
                    image_path = (PROJECT_ROOT / image_path).resolve()
                self.samples.append((image_path, int(label_text)))

        if not self.samples:
            raise ValueError(f"{self.csv_path} contains no image rows.")

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
