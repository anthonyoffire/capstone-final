import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from dataset import AgeDataset
from model import build_model
from image_shuffler import PIL_Shuffler, validate_grid_size

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE

IMAGE_SIZE = 128
BATCH_SIZE = 32
GRID_SIZE = 1

def get_transforms():
    return transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    PIL_Shuffler(grid_size=GRID_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(model_p):
    save_model = torch.load(model_p, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(save_model["model_state_dict"])
    model.eval()
    return model


def eval(csv_path, modelp, threshold=18, output_dir="./eval_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    trans = get_transforms()
    evaluation_dataset = AgeDataset(csv_path, transform=trans)

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = load_model(modelp)

    actual = []
    predicted = []

    with torch.no_grad():
        for images, ages in tqdm(loader, desc="evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)
            preds = model(images).cpu().numpy().flatten().tolist()
            act = (ages * AGE_RANGE + MIN_AGE).cpu().numpy().flatten()
            actual.extend(act.tolist())
            predicted.extend(preds)
    actual = np.array(actual)
    predicted = np.array(predicted)

    summary_rows = []
    per_age_cls_rows = []

    true_cls = actual >= threshold
    pred_cls = predicted >= 0.5

    tp = int(np.sum(pred_cls & true_cls))
    tn = int(np.sum((~pred_cls) & (~true_cls)))
    fp = int(np.sum(pred_cls & (~true_cls)))
    fn = int(np.sum((~pred_cls) & true_cls))

    total = len(actual)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n=== Classification @ {threshold} ===")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    summary_rows.append([
        threshold, accuracy, precision, recall, specificity, f1, tp, tn, fp, fn
    ])

    # per-age classification
    for age in sorted(set(int(round(x)) for x in actual)):
        mask = np.round(actual).astype(int) == age
        if np.sum(mask) == 0:
            continue

        age_true = actual[mask]
        age_pred = predicted[mask]

        age_acc = np.mean((age_true >= threshold) == (age_pred >= 0.5))

        per_age_cls_rows.append([
            threshold, age, np.sum(mask), age_acc
        ])

    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "accuracy", "precision", "recall",
            "specificity", "f1", "tp", "tn", "fp", "fn"
        ])
        writer.writerows(summary_rows)

    print(f"Saved: {summary_path}")

    per_age_cls_path = os.path.join(output_dir, "per_age_classification.csv")
    with open(per_age_cls_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "age", "count", "accuracy"])
        writer.writerows(per_age_cls_rows)

    print(f"Saved: {per_age_cls_path}")

if __name__ == '__main__':
    csv_path = "./synthetic_csv/uniformTest.csv"
    model_path = "./best_models_synth/sam90u18.pt"
    parser = argparse.ArgumentParser(description="Shuffle images in a directory using a square grid.")
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=1, # No scramble
        help="Grid size (e.g., 4 creates a 4x4 grid). Must be a power of 2 <= 128, > 0"
    )
    args = parser.parse_args()
    GRID_SIZE = args.grid_size
    eval(csv_path, model_path, threshold=18, output_dir="./eval_outputs_u18_sam90")
