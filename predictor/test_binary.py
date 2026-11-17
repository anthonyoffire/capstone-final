import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import faulthandler
faulthandler.enable()

import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path

from dataset import AgeDataset
from model import build_model
from image_shuffler import PIL_Shuffler, validate_grid_size
from batch_transforms import BatchImageTransforms
from confusion_matrix import print_ascii_confusion_matrix, save_confusion_matrix
from evaluation_metrics import wilson_confidence_interval

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE

IMAGE_SIZE = 128
BATCH_SIZE = 32
GRID_SIZE = 1
MODEL_NAME = "efficientnet_b0"
MODELS = [f"efficientnet_b{index}" for index in range(8)] + [
    "resnet50",
    "densenet121",
    "convnext_tiny",
]
PROBABILITY_THRESHOLD = 0.5
VAL_TUNING_THRESHOLDS = [x / 100 for x in range(40, 81, 5)]
OUTPUT_DIR = Path("./eval_outputs_binary/s2s_move_shuffle")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE_DIR = PROJECT_ROOT / "best_models_bin"
CSV_BASE_DIR = PROJECT_ROOT / "synthetic_csv"

# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

def _resolve_input_path(path: Path, base_dir: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].casefold() == base_dir.name.casefold():
        return (base_dir.parent / path).resolve()
    return (base_dir / path).resolve()


def get_transforms():
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

def load_model(model_p):
    save_model = torch.load(model_p, map_location=DEVICE)
    model = build_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(save_model["model_state_dict"])
    model.eval()
    return model


def write_tuning_markdown(report_path, checkpoint_name, csv_path, age_threshold, rows):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Binary threshold tuning report",
        "",
        f"- **Checkpoint:** `{checkpoint_name}`",
        f"- **Validation CSV:** `{csv_path}`",
        f"- **Age cutoff:** {age_threshold}",
        "",
    ]

    for row in rows:
        (
            row_age_threshold,
            probability_threshold,
            accuracy,
            precision,
            recall,
            specificity,
            specificity_ci_lower,
            specificity_ci_upper,
            f1,
            tp,
            tn,
            fp,
            fn,
        ) = row
        actual_under_total = tn + fp
        actual_over_total = fn + tp
        predicted_under_total = tn + fn
        predicted_over_total = fp + tp
        total = actual_under_total + actual_over_total

        lines.extend([
            f"## Probability threshold: {probability_threshold:g}",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Accuracy | {accuracy:.2%} |",
            f"| Precision | {precision:.2%} |",
            f"| Recall | {recall:.2%} |",
            f"| Specificity | {specificity:.2%} |",
            f"| Specificity 95% CI | {specificity_ci_lower:.2%}-{specificity_ci_upper:.2%} |",
            f"| F1 score | {f1:.2%} |",
            "",
            "| Predicted / Actual "
            f"| Actual < {row_age_threshold} "
            f"| Actual >= {row_age_threshold} | Predicted total |",
            "|---|---:|---:|---:|",
            f"| Predicted < {row_age_threshold} | TN: {tn:,} "
            f"| FN: {fn:,} | {predicted_under_total:,} |",
            f"| Predicted >= {row_age_threshold} | FP: {fp:,} "
            f"| TP: {tp:,} | {predicted_over_total:,} |",
            f"| Actual total | {actual_under_total:,} "
            f"| {actual_over_total:,} | {total:,} |",
            "",
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def eval(
    csv_path,
    modelp,
    age_threshold,
    probability_thresholds=None,
    output_dir: str | Path = "./eval_outputs",
    save_outputs=True,
    markdown_report_path=None,
    run_label: str | None = None,
):
    if probability_thresholds is None:
        probability_thresholds = [PROBABILITY_THRESHOLD]
    probability_thresholds = tuple(probability_thresholds)
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare the data and model.
    use_cuda_transforms = DEVICE == "cuda"
    trans = get_gpu_input_transforms() if use_cuda_transforms else get_transforms()
    batch_transforms = None
    if use_cuda_transforms:
        batch_transforms = BatchImageTransforms(
            grid_size=GRID_SIZE,
            device=DEVICE,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    evaluation_dataset = AgeDataset(csv_path, transform=trans)

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    model = load_model(modelp)

    actual = []
    predicted = []

    # Run inference.
    with torch.no_grad():
        for images, ages, _ in tqdm(loader, desc="evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            if batch_transforms is not None:
                images = batch_transforms(images)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)
            preds = model(images).cpu().numpy().flatten().tolist()
            act = (ages * AGE_RANGE + MIN_AGE).cpu().numpy().flatten()
            actual.extend(act.tolist())
            predicted.extend(preds)
    actual = np.array(actual)
    predicted = np.array(predicted)
    true_cls = actual >= age_threshold

    summary_rows = []
    per_age_cls_rows = []

    # Evaluate predictions against the requested age boundary.
    total = len(actual)
    for probability_threshold in probability_thresholds:
        pred_cls = predicted >= probability_threshold

        tp = int(np.sum(pred_cls & true_cls))
        tn = int(np.sum((~pred_cls) & (~true_cls)))
        fp = int(np.sum(pred_cls & (~true_cls)))
        fn = int(np.sum((~pred_cls) & true_cls))

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_ci_lower, specificity_ci_upper = wilson_confidence_interval(
            tn, tn + fp
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(
            f"\n=== Classification @ age {age_threshold}, "
            f"probability {probability_threshold:g} ==="
        )
        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}"
        )
        print(
            f"Specificity 95% CI: "
            f"{specificity_ci_lower:.4f}-{specificity_ci_upper:.4f}"
        )
        print_ascii_confusion_matrix(
            age_threshold,
            probability_threshold,
            tn,
            fp,
            fn,
            tp,
            run_label or f"checkpoint {Path(modelp).stem}",
        )

        summary_rows.append([
            age_threshold, probability_threshold, accuracy, precision, recall, specificity,
            specificity_ci_lower, specificity_ci_upper, f1, tp, tn, fp, fn
        ])

        if save_outputs:
            matrix_path = save_confusion_matrix(
                output_dir,
                age_threshold,
                tn,
                fp,
                fn,
                tp,
                accuracy,
                recall,
                specificity,
            )
            print(f"Saved confusion matrix: {matrix_path}")

            for age in sorted(set(int(round(x)) for x in actual)):
                mask = np.round(actual).astype(int) == age
                if np.sum(mask) == 0:
                    continue

                age_true = true_cls[mask]
                age_pred = predicted[mask] >= probability_threshold
                age_acc = np.mean(age_true == age_pred)
                per_age_cls_rows.append([
                    probability_threshold, age, np.sum(mask), age_acc
                ])

    if save_outputs:
        summary_path = os.path.join(output_dir, "summary_metrics.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "age_threshold", "probability_threshold", "accuracy", "precision", "recall",
                "specificity", "specificity_95ci_lower",
                "specificity_95ci_upper", "f1", "tp", "tn", "fp", "fn"
            ])
            writer.writerows(summary_rows)

        print(f"Saved: {summary_path}")

        per_age_cls_path = os.path.join(output_dir, "per_age_classification.csv")
        with open(per_age_cls_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "probability_threshold", "age", "count", "accuracy"
            ])
            writer.writerows(per_age_cls_rows)

        print(f"Saved: {per_age_cls_path}")

    if markdown_report_path is not None:
        report_path = write_tuning_markdown(
            markdown_report_path,
            Path(modelp).stem,
            csv_path,
            age_threshold,
            summary_rows,
        )
        print(f"Saved threshold tuning report: {report_path}")

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate binary classifier checkpoints against a labeled CSV."
    )
    parser.add_argument(
        "-g", "--grid_size",
        type=validate_grid_size,
        default=GRID_SIZE,
        help="Positive grid size; both image dimensions must be divisible by it.",
    )
    parser.add_argument(
        "--age",
        type=int,
        required=True,
        help="Age boundary used to define class 1 (ages at or above this value).",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=MODEL_NAME,
        help="Torchvision model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint; relative filenames resolve under MODEL_BASE_DIR.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base directory for evaluation outputs.",
    )
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument(
        "-t", "--test_csv",
        type=Path,
        help="Test CSV; relative filenames resolve under CSV_BASE_DIR.",
    )
    csv_group.add_argument(
        "--validation_csv",
        type=Path,
        help="Validation CSV; relative filenames resolve under CSV_BASE_DIR and tune the threshold.",
    )
    args = parser.parse_args()
    evaluation_csv = _resolve_input_path(
        args.validation_csv or args.test_csv,
        CSV_BASE_DIR,
    )
    if not evaluation_csv.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {evaluation_csv}")
    model_path = _resolve_input_path(args.checkpoint, MODEL_BASE_DIR)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    GRID_SIZE = args.grid_size
    MODEL_NAME = args.model
    run_label = f"checkpoint {model_path.stem}"
    print(f"\n=== Evaluating {run_label} ===")
    checkpoint_name = model_path.stem
    threshold_tuning = args.validation_csv is not None
    if threshold_tuning:
        probability_thresholds = VAL_TUNING_THRESHOLDS
        output_dir = args.output_dir / (
            f"{checkpoint_name}_{evaluation_csv.stem}_age{args.age}_g{GRID_SIZE}_tuning"
        )
        markdown_report_path = output_dir / "threshold_tuning_report.md"
    else:
        probability_thresholds = [PROBABILITY_THRESHOLD]
        output_dir = args.output_dir / (
            f"{checkpoint_name}_{evaluation_csv.stem}_age{args.age}_g{GRID_SIZE}"
        )
        markdown_report_path = None
    eval(
        evaluation_csv,
        model_path,
        age_threshold=args.age,
        probability_thresholds=probability_thresholds,
        output_dir=output_dir,
        save_outputs=not threshold_tuning,
        markdown_report_path=markdown_report_path,
        run_label=run_label,
    )
