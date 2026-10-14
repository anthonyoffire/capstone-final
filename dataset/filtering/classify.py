import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from model import CLASSIFICATION_THRESHOLD, build_classifier, get_image_transform

NUM_WORKERS = 2
BATCH_SIZE = 32


def is_image(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_preserve_structure(base_path: Path, src_path: Path, dst_root: Path):
    rel = src_path.relative_to(base_path)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, index, None
        except Exception as error:
            return None, index, str(error)


def collate_image_batch(batch):
    tensors = [tensor for tensor, _, error in batch if error is None]
    indices = [index for _, index, _ in batch]
    errors = [(index, error) for _, index, error in batch if error is not None]
    image_batch = torch.stack(tensors) if tensors else None
    return image_batch, indices, errors


def classify_images(input_dir: Path, output_dir: Path, model_dir: str, subdir0: str, subdir1: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(model_dir), map_location=device)

    print(f"Using device: {device}")
    
    model_type = ckpt["model_type"]
    model = build_classifier(model_type).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_image_transform(model_type)

    image_paths = [p for p in input_dir.rglob("*") if p.is_file() and is_image(p)]
    image_dataset = ImagePathDataset(image_paths, transform)
    data_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate_image_batch,
    )

    with torch.no_grad():
        progress = tqdm(total=len(image_paths), desc=f"Classifying ({model_type})", unit="image")
        for tensors, indices, errors in data_loader:
            for index, error in errors:
                tqdm.write(f"Skipped {image_paths[index]}: {error}")

            if tensors is not None:
                probabilities = model(tensors.to(device, non_blocking=True)).view(-1).cpu()
                valid_indices = [
                    index for index in indices
                    if all(error_index != index for error_index, _ in errors)
                ]
                for index, probability in zip(valid_indices, probabilities):
                    destination_root = output_dir / (
                        subdir1 if probability.item() >= CLASSIFICATION_THRESHOLD else subdir0
                    )
                    try:
                        copy_preserve_structure(input_dir, image_paths[index], destination_root)
                    except Exception as error:
                        tqdm.write(f"Skipped {image_paths[index]}: {error}")

            progress.update(len(indices))
        progress.close()

    print("Done.")
