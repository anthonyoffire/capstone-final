import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import faulthandler
faulthandler.enable()

import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import random
from multiprocessing.spawn import freeze_support
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from model import build_model
from image_shuffler import PIL_Shuffler, validate_grid_size
from batch_transforms import BatchImageTransforms
from binary_age_dataset import BinaryAgeDataset

IMAGE_SIZE = 128
GRID_SIZE = 1 # grid size x: x by x grid
MODEL_NAME = "efficientnet_b0"
BATCH_SIZE = 128
NUM_EPOCHS = 20
MIN_EPOCHS = 1
PATIENCE = 5
MIN_DELTA = 0.0001
NUM_WORKERS = 4
PIN_MEMORY = False
CHECKPOINT_DIR = Path("./best_models_bin2")
CSV_BASE_DIR = Path("./synthetic_csv")

M_SHORT_NAMES = {
    "efficientnet_b0": "enb0"
}

def get_training_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        PIL_Shuffler(grid_size=GRID_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


def get_validation_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        PIL_Shuffler(grid_size=GRID_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


def get_gpu_input_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.PILToTensor(),
    ])

def main(
    training_csv: Path,
    validation_csv: Path,
):
    training_csv = CSV_BASE_DIR / training_csv
    validation_csv = CSV_BASE_DIR / validation_csv
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    num_channels = 3
    num_gpus = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

    batch_transforms = None
    if device.type == "cuda":
        train_transform = get_gpu_input_transforms()
        val_transform = get_gpu_input_transforms()
        batch_transforms = BatchImageTransforms(
            grid_size=GRID_SIZE,
            image_size=IMAGE_SIZE,
            device=device,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        print("Image shuffling and augmentation will run on CUDA.")
    else:
        train_transform = get_training_transforms()
        val_transform = get_validation_transforms()

    train_dataset = BinaryAgeDataset(training_csv, transform=train_transform)
    validation_dataset = BinaryAgeDataset(validation_csv, transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError("No training images were found.")
    if len(validation_dataset) == 0:
        raise ValueError("No validation images were found.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    classifier = build_model(MODEL_NAME).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())

    train_source_name = Path(training_csv).stem
    short_name = M_SHORT_NAMES[MODEL_NAME]
    checkpoint_path = CHECKPOINT_DIR / f"{short_name}_{train_source_name}_g{GRID_SIZE}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        train_loss_sum = 0.0
        train_count = 0
        with tqdm(
            total=len(train_dataset),
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} train",
            unit="img",
        ) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                if batch_transforms is not None:
                    inputs = batch_transforms(inputs, augment=True)
                labels = labels.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                output = classifier(inputs).view(-1)
                err = criterion(output, labels)
                batch_loss = err.item()
                train_loss_sum += batch_loss * inputs.size(0)
                train_count += inputs.size(0)
                err.backward()
                optimizer.step()
                pbar.update(inputs.size(0))
                pbar.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss = train_loss_sum / train_count
        print(f"Train loss: {train_loss:.6f}")

        classifier.eval()
        val_loss_sum = 0.0
        val_count = 0
        with tqdm(
            total=len(validation_dataset),
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} val",
            unit="img",
        ) as pbar:
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    if batch_transforms is not None:
                        inputs = batch_transforms(inputs)
                    labels = labels.to(device=device, dtype=torch.float32)
                    output = classifier(inputs).view(-1)
                    err = criterion(output, labels)
                    batch_loss = err.item()
                    val_loss_sum += batch_loss * inputs.size(0)
                    val_count += inputs.size(0)
                    pbar.update(inputs.size(0))
                    pbar.set_postfix(loss=f"{batch_loss:.4f}")

        val_loss = val_loss_sum / val_count
        print(f"Validation loss: {val_loss:.6f}")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "num_channels": num_channels,
                },
                checkpoint_path,
            )
            print(
                f"Saved best validation checkpoint to {checkpoint_path} "
                f"(loss={val_loss:.6f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{PATIENCE} epoch(s)."
            )
            if (
                epoch + 1 >= MIN_EPOCHS
                and epochs_without_improvement >= PATIENCE
            ):
                print(f"Early stopping after epoch {epoch + 1}.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a binary age classifier from labeled CSV manifests."
    )
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=GRID_SIZE,
        help="Grid size (e.g., 4 creates a 4x4 grid). Must be a power of 2 <= 128, > 0"
    )
    parser.add_argument(
        "-t",
        "--training_csv",
        type=Path,
        required=True,
        help="Training CSV; relative paths resolve from CSV_BASE_DIR.",
    )
    parser.add_argument(
        "-v",
        "--validation_csv",
        type=Path,
        required=True,
        help="Validation CSV; relative paths resolve from CSV_BASE_DIR.",
    )
    args = parser.parse_args()
    GRID_SIZE = args.grid_size
    freeze_support()
    main(
        training_csv=args.training_csv,
        validation_csv=args.validation_csv,
    )