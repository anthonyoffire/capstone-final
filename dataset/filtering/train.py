import random
from multiprocessing.spawn import freeze_support
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset

from model import (
    CLASSIFICATION_THRESHOLD,
    LEARNING_RATES,
    MODEL_TYPES,
    build_classifier,
    get_image_transform,
)

MAX_MISCLASS_PERCENT = 26.0
MAX_VALLOSS_INCREASE = 1.1

def train_model(model_type):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    dataroot = "dataset/sort"
    batch_size = 32
    max_epochs = 30
    validation_fraction = 0.3
    min_delta = 1e-5
    workers = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Training model: {model_type}")

    dataset = dset.ImageFolder(root=dataroot, transform=get_image_transform(model_type))
    bad_class_index = dataset.class_to_idx["bad"]
    good_class_index = dataset.class_to_idx["good"]
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for class_index in range(len(dataset.classes)):
        class_indices = [index for index, target in enumerate(dataset.targets) if target == class_index]
        shuffled = torch.randperm(len(class_indices), generator=generator).tolist()
        validation_count = max(1, round(len(class_indices) * validation_fraction))
        validation_indices.extend(class_indices[index] for index in shuffled[:validation_count])
        train_indices.extend(class_indices[index] for index in shuffled[validation_count:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    classifier = build_classifier(
        model_type,
        pretrained=True,
    ).to(device)
    criterion = nn.BCELoss()
    learning_rate = LEARNING_RATES[model_type]
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    ckpt_path = Path("checkpoints") / f"classifier_{model_type}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_error = float("inf")
    minimum_validation_loss = float("inf")
    for epoch in range(max_epochs):
        classifier.train()
        training_loss_total = 0.0
        training_sample_count = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            classifier.zero_grad()
            output = classifier(inputs).view(-1)
            err = criterion(output, labels)
            training_loss_total += err.item() * inputs.size(0)
            training_sample_count += inputs.size(0)
            err.backward()
            optimizer.step()
        training_loss = training_loss_total / training_sample_count

        classifier.eval()
        validation_loss_total = 0.0
        validation_sample_count = 0
        good_misclassified_count = 0
        bad_misclassified_count = 0
        actual_good_count = 0
        actual_bad_count = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                output = classifier(inputs).view(-1)
                batch_loss = criterion(output, labels)
                validation_loss_total += batch_loss.item() * inputs.size(0)
                validation_sample_count += inputs.size(0)
                predicted_good = output >= CLASSIFICATION_THRESHOLD
                actual_good = labels == good_class_index
                actual_bad = labels == bad_class_index
                good_misclassified_count += ((~predicted_good) & actual_good).sum().item()
                bad_misclassified_count += (predicted_good & actual_bad).sum().item()
                actual_good_count += actual_good.sum().item()
                actual_bad_count += actual_bad.sum().item()
        validation_loss = validation_loss_total / validation_sample_count
        minimum_validation_loss = min(minimum_validation_loss, validation_loss)
        stop_for_validation_loss = validation_loss > minimum_validation_loss * MAX_VAL_LOSS_INCREASE
        good_misclassified_percent = 100 * good_misclassified_count / actual_good_count
        bad_misclassified_percent = 100 * bad_misclassified_count / actual_bad_count
        validation_average_class_error = (
            good_misclassified_percent + bad_misclassified_percent
        ) / 2
        checkpoint_eligible = (
            epoch < 2 or (
                good_misclassified_percent < MAX_MISCLASS_PERCENT
                and bad_misclassified_percent < MAX_MISCLASS_PERCENT
            )
        )
        print(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"train_loss={training_loss:.5f}, "
            f"val_loss={validation_loss:.5f}, "
            f"good_misclassified={good_misclassified_percent:.2f}% "
            f"({good_misclassified_count}/{actual_good_count}), "
            f"bad_misclassified={bad_misclassified_percent:.2f}% "
            f"({bad_misclassified_count}/{actual_bad_count}), "
            f"ave_class_error={validation_average_class_error:.2f}%, "
        )

        if checkpoint_eligible and validation_average_class_error / 100 < best_validation_error - min_delta:
            best_validation_error = validation_average_class_error / 100
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": validation_loss,
                    "best_average_class_error": best_validation_error,
                    "model_type": model_type,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(
                f"  Saved best checkpoint to {ckpt_path} "
                f"(average class error={validation_average_class_error:.2f}%)"
            )

        if stop_for_validation_loss:
            print(
                f"Early stopping after {epoch + 1} epochs; val_loss={validation_loss:.5f} "
            )
            break

def main():
    for model_type in MODEL_TYPES:
        train_model(model_type)


if __name__ == '__main__':
    freeze_support()
    main()