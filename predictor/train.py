import os
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
from image_shuffler import PickleShuffler, validate_grid_size

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

CSV_PATH = "./synthetic_csv/styleganPopulation80.csv"
SAVE_PATH = "best_models_synth/styleganPopulation80.pth"

GRID_SIZE = 1

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        PickleShuffler(grid_size=GRID_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def run():
    print("Device:", DEVICE)

    train_tf = get_transforms()

    train_ds = AgeDataset(CSV_PATH, transform=train_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = build_model().to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (DEVICE == "cuda")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        model.train()
        running_loss = 0.0

        for images, ages in tqdm(train_loader, desc="train"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, ages)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss (normalized MAE): {train_loss:.4f}")

        scheduler.step()
        print("LR:", scheduler.get_last_lr()[0])

    torch.save({
    "model_state_dict": model.state_dict(),
    "image_size": IMAGE_SIZE,
    "max_age": MAX_AGE,
    "min_age": MIN_AGE,},
    SAVE_PATH)
    print(f"Saved best checkpoint -> {SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle images in a directory using a square grid.")
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=1, # No scramble
        help="Grid size (e.g., 4 creates a 4x4 grid). Must be a power of 2 <= 128, > 0"
    )
    args = parser.parse_args()
    GRID_SIZE = args.grid_size
    run()