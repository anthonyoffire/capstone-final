import faulthandler
faulthandler.enable()

import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import os
import traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

from dataset import AgeDataset
from model import build_model
from image_shuffler import PIL_Shuffler, validate_grid_size
from batch_transforms import BatchImageTransforms



IMAGE_SIZE = 128
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 0.0001
LR = 3e-4
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True
LOAD_CHECKPOINT_IF_EXISTS = True

CSV_DIR = "./synthetic_csv/"
SAVE_DIR = "best_models/"
TRAIN_CSV = ""
VAL_CSV = ""
SAVE_PATH = ""
GRID_SIZE = 1

MODEL = "efficientnet_b0"
MODELS = [
    # "resnet18",
    # "resnet34",
    # "resnet50",
    # "densenet121",
    # "densenet169",Y
    # "densenet201",
    # "efficientnet_b3",
    # "efficientnet_v2_s",
    # "convnext_tiny",
    "efficientnet_b0"
]
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


def run():
    print("Device:", DEVICE)

    batch_transforms = None
    if DEVICE == "cuda":
        train_tf = get_gpu_input_transforms()
        validation_tf = get_gpu_input_transforms()
        batch_transforms = BatchImageTransforms(
            grid_size=GRID_SIZE,
            image_size=IMAGE_SIZE,
            device=DEVICE,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        print("Image shuffling and augmentation will run on CUDA.")
    else:
        train_tf = get_training_transforms()
        validation_tf = get_validation_transforms()

    train_ds = AgeDataset(TRAIN_CSV, transform=train_tf)
    validation_ds = AgeDataset(VAL_CSV, transform=validation_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    print(len(MODELS))
    for model_name in MODELS:
        try:
            model = build_model(model_name).to(DEVICE)
            SAVE_PATH = SAVE_DIR + f"{model_name}_{os.path.basename(TRAIN_CSV).removesuffix('.csv')}_g{GRID_SIZE}.pth"
            if LOAD_CHECKPOINT_IF_EXISTS:
                if os.path.isfile(SAVE_PATH):
                    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"Loaded checkpoint weights from {SAVE_PATH}")
                else:
                    print(f"No checkpoint found at {SAVE_PATH}; starting with fresh weights.")
            criterion = nn.L1Loss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

            use_amp = (DEVICE == "cuda")

            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best_val_loss = float("inf")
            epochs_without_improvement = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                print(f"\nEpoch {epoch}")

                model.train()
                running_loss = 0.0

                total_images = len(train_ds)
                with tqdm(total=total_images, desc="train", unit="img") as pbar:
                    for images, ages, _ in train_loader:
                        batch_len = len(images)
                        images = images.to(DEVICE, non_blocking=True)
                        if batch_transforms is not None:
                            images = batch_transforms(images, augment=True)
                        ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            preds = model(images)
                            loss = criterion(preds, ages)

                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()

                        running_loss += loss.item()
                        # Update image count and display in progress bar
                        pbar.update(batch_len)
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                train_loss = running_loss / max(1, len(train_loader))
                print(f"Train loss (normalized MAE): {train_loss:.4f}")

                # Validation
                model.eval()
                val_running_loss = 0.0
                with tqdm(total=len(validation_ds), desc="val", unit="img") as pbar:
                    with torch.no_grad():
                        for images, ages, _ in validation_loader:
                            batch_len = len(images)
                            images = images.to(DEVICE, non_blocking=True)
                            if batch_transforms is not None:
                                images = batch_transforms(images)
                            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

                            with torch.amp.autocast("cuda", enabled=use_amp):
                                preds = model(images)
                                loss = criterion(preds, ages)

                            val_running_loss += loss.item()
                            pbar.update(batch_len)
                            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                val_loss = val_running_loss / max(1, len(validation_loader))
                print(f"Validation loss: {val_loss:.4f}")

                print("LR:", scheduler.get_last_lr()[0])

                scheduler.step()

                if val_loss < best_val_loss - MIN_DELTA and val_loss < .153:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "image_size": IMAGE_SIZE,
                        "max_age": MAX_AGE,
                        "min_age": MIN_AGE,
                    }, SAVE_PATH)
                    print(f"Saved new best checkpoint -> {SAVE_PATH} (Val Loss: {val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"No significant improvement for {epochs_without_improvement}/{PATIENCE} epoch(s).")
                    if epochs_without_improvement >= PATIENCE:
                        print(f"Early stopping after epoch {epoch}.")
                        break
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle images in a directory using a square grid.")
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=128, # No scramble
        help="Grid size (e.g., 4 creates a 4x4 grid). Must be a power of 2 <= 128, > 0"
    )
    parser.add_argument(
        "-t", "--training_csv",
        required=False,
        type=str,
        default="serenity8020_train.csv",
        help="Csv file for train dataset"
    )
    parser.add_argument(
        "-v", "--validation_csv",
        required=False,
        type=str,
        default="serenity8020_val.csv",
        help="Csv file for validation dataset"
    )
    args = parser.parse_args()
    GRID_SIZE = args.grid_size
    TRAIN_CSV = CSV_DIR + args.training_csv
    VAL_CSV = CSV_DIR + args.validation_csv
    SAVE_PATH = SAVE_DIR + MODEL + "_" + args.training_csv.removesuffix(".csv") + f"_g{GRID_SIZE}.pth"
    run()