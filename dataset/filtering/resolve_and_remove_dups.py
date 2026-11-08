import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from predictor.model import build_model

INPUT_CSV = Path(__file__).absolute().with_name("duplicates.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIVOLO_DIR = Path.home() / "MiVOLO"

MODEL_PATHS = {
    "mivolo": {
        "checkpoint": (
            PROJECT_ROOT
            / "dataset"
            / "generation"
            / "pretrained_models"
            / "model_imdb_cross_person_4.22_99.46.pth.tar"
        ),
    },
    "resnet18": PROJECT_ROOT / "best_models" / "resnet18_train_g1.pth",
    "efficientnet_b4": (
        PROJECT_ROOT / "best_models" / "efficientnet_b4_realonly.pth"
    ),
}
MODEL_ORDER = ("mivolo", "resnet18", "efficientnet_b4")
BATCH_SIZE = 32

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
IMAGE_SIZE = 128
AGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_hash_groups(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    groups = defaultdict(dict)
    path_hashes = {}
    required_columns = {"hash", "file_path", "age"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            hash_value = (row["hash"] or "").strip()
            file_path = row["file_path"] or ""
            age_text = (row["age"] or "").strip()
            if not hash_value or not file_path or not age_text:
                raise ValueError(
                    f"CSV row {row_number} has an empty hash, file_path, or age."
                )

            try:
                age_value = float(age_text)
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {row_number} has an invalid age: {age_text!r}"
                ) from exc
            if not math.isfinite(age_value) or not age_value.is_integer():
                raise ValueError(
                    f"CSV row {row_number} has a non-integer age: {age_text!r}"
                )

            age = int(age_value)
            previous_hash = path_hashes.get(file_path)
            if previous_hash is not None and previous_hash != hash_value:
                raise ValueError(
                    f"CSV assigns multiple hashes to {file_path}: "
                    f"{previous_hash} and {hash_value}"
                )
            path_hashes[file_path] = hash_value
            existing_age = groups[hash_value].get(file_path)
            if existing_age is not None and existing_age != age:
                raise ValueError(
                    f"CSV assigns multiple ages to {file_path}: "
                    f"{existing_age} and {age}"
                )
            groups[hash_value][file_path] = age

    if not groups:
        raise ValueError(f"No image rows found in CSV: {csv_path}")
    return groups


def validate_model_paths():
    if not MIVOLO_DIR.is_dir():
        raise FileNotFoundError(f"MiVOLO checkout not found: {MIVOLO_DIR}")

    for model_name in ("resnet18", "efficientnet_b4"):
        model_path = MODEL_PATHS[model_name]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_name} checkpoint not found: {model_path}"
            )

    mivolo_checkpoint = MODEL_PATHS["mivolo"]["checkpoint"]
    if not mivolo_checkpoint.is_file():
        raise FileNotFoundError(
            f"MiVOLO checkpoint not found: {mivolo_checkpoint}"
        )


def load_regression_model(model_name, model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_regression_ages(images, model):
    image_batch = torch.stack([AGE_TRANSFORM(image) for image in images]).to(DEVICE)
    with torch.inference_mode():
        normalized_ages = model(image_batch).reshape(-1).cpu().tolist()
    return [
        normalized_age * AGE_RANGE + MIN_AGE
        for normalized_age in normalized_ages
    ]


def predict_mivolo_ages(images_bgr, model, prepare_classification_images):
    if not model.meta.with_persons_model or not model.meta.disable_faces:
        raise ValueError(
            "Direct MiVOLO image batching requires a person model with "
            "face crops disabled."
        )

    batch_size = len(images_bgr)
    face_batch = prepare_classification_images(
        [None] * batch_size,
        target_size=model.input_size,
        mean=model.data_config["mean"],
        std=model.data_config["std"],
        device=model.device,
    )
    person_batch = prepare_classification_images(
        images_bgr,
        target_size=model.input_size,
        mean=model.data_config["mean"],
        std=model.data_config["std"],
        device=model.device,
    )
    model_input = torch.cat((face_batch, person_batch), dim=1)
    with torch.inference_mode():
        output = model.inference(model_input)

    if model.meta.only_age:
        normalized_ages = output.reshape(batch_size, -1)[:, 0]
    else:
        normalized_ages = output[:, 2]
    ages = (
        normalized_ages * (model.meta.max_age - model.meta.min_age)
        + model.meta.avg_age
    )
    return [float(age) for age in ages.cpu().tolist()]


def load_estimators():
    validate_model_paths()

    sys.path.insert(0, str(MIVOLO_DIR))
    from mivolo.data.misc import prepare_classification_images
    from mivolo.model.mi_volo import MiVOLO

    mivolo_model = MiVOLO(
        str(MODEL_PATHS["mivolo"]["checkpoint"]),
        device="cuda:0" if DEVICE == "cuda" else "cpu",
        half=True,
        use_persons=True,
        disable_faces=True,
        verbose=False,
    )

    resnet18 = load_regression_model(
        "resnet18", MODEL_PATHS["resnet18"]
    )
    efficientnet_b4 = load_regression_model(
        "efficientnet_b4", MODEL_PATHS["efficientnet_b4"]
    )

    return {
        "mivolo": lambda images: predict_mivolo_ages(
            images, mivolo_model, prepare_classification_images
        ),
        "resnet18": lambda images: predict_regression_ages(images, resnet18),
        "efficientnet_b4": lambda images: predict_regression_ages(
            images, efficientnet_b4
        ),
    }


def choose_winners(groups, estimators):
    winners = {}
    ambiguous_hashes = []
    classification_errors = []

    group_items = list(groups.items())
    with tqdm(
        total=len(group_items), desc="Classifying hashes", unit="hash"
    ) as pbar:
        for start in range(0, len(group_items), BATCH_SIZE):
            batch_items = group_items[start : start + BATCH_SIZE]
            loaded = []
            for hash_value, path_ages in batch_items:
                image_path = next(iter(path_ages))
                try:
                    with Image.open(image_path) as source_image:
                        image = source_image.convert("RGB")
                    image_bgr = np.asarray(image)[:, :, ::-1].copy()
                except Exception as exc:
                    classification_errors.append(
                        (hash_value, "image loading", image_path, str(exc))
                    )
                    print(
                        f"image loading failed for hash {hash_value} "
                        f"({image_path}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                loaded.append(
                    (hash_value, path_ages, image_path, image, image_bgr)
                )
            pbar.update(len(batch_items))
            if not loaded:
                continue

            batch_predictions = [[] for _ in loaded]
            for model_name in MODEL_ORDER:
                images = [
                    item[4] if model_name == "mivolo" else item[3]
                    for item in loaded
                ]
                try:
                    predictions = estimators[model_name](images)
                    if len(predictions) != len(loaded):
                        raise ValueError(
                            f"expected {len(loaded)} predictions, got "
                            f"{len(predictions)}"
                        )
                except Exception as exc:
                    for hash_value, _, image_path, _, _ in loaded:
                        classification_errors.append(
                            (hash_value, model_name, image_path, str(exc))
                        )
                        print(
                            f"{model_name} failed for hash {hash_value} "
                            f"({image_path}): {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue

                for index, (hash_value, _, image_path, _, _) in enumerate(loaded):
                    try:
                        prediction = float(predictions[index])
                        if not math.isfinite(prediction):
                            raise ValueError(
                                "estimator returned a non-finite age"
                            )
                    except Exception as exc:
                        classification_errors.append(
                            (hash_value, model_name, image_path, str(exc))
                        )
                        print(
                            f"{model_name} failed for hash {hash_value} "
                            f"({image_path}): {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    batch_predictions[index].append(prediction)

            for (hash_value, path_ages, image_path, _, _), predictions in zip(
                loaded, batch_predictions
            ):
                if not predictions:
                    continue

                average_age = sum(predictions) / len(predictions)
                candidate_ages = sorted(set(path_ages.values()))
                distances = {
                    age: abs(average_age - age)
                    for age in candidate_ages
                }
                closest_distance = min(distances.values())
                closest_ages = [
                    age
                    for age, distance in distances.items()
                    if math.isclose(
                        distance, closest_distance, rel_tol=0.0, abs_tol=1e-9
                    )
                ]
                if len(closest_ages) != 1:
                    ambiguous_hashes.append(hash_value)
                    continue

                winning_age = closest_ages[0]
                winning_path = next(
                    path for path, age in path_ages.items()
                    if age == winning_age
                )
                winners[hash_value] = {
                    "path": winning_path,
                    "age": winning_age,
                    "losers": [
                        path for path in path_ages if path != winning_path
                    ],
                }

    return winners, ambiguous_hashes, classification_errors


def main():
    csv_path = INPUT_CSV
    groups = load_hash_groups(csv_path)
    estimators = load_estimators()
    winners, ambiguous_hashes, classification_errors = choose_winners(
        groups, estimators
    )

    unique_hash_count = len(groups)
    winner_count = len(winners)
    singular_age_count = winner_count
    delete_count = sum(len(result["losers"]) for result in winners.values())
    calculation_succeeded = (
        not ambiguous_hashes and winner_count == unique_hash_count
    )
    winner_count_matches = winner_count == unique_hash_count

    print("\nDuplicate resolution summary")
    print(f"Unique hashes: {unique_hash_count}")
    print(f"Hashes with one closest age: {singular_age_count}")
    print(f"Singular closest-age calculation successful: {calculation_succeeded}")
    print(f"Winning images selected: {winner_count}")
    print(f"Winning images match unique hashes: {winner_count_matches}")
    print(f"Ambiguous closest-age ties: {len(ambiguous_hashes)}")
    print(f"Estimator failures: {len(classification_errors)}")
    print(f"Images scheduled for deletion: {delete_count}")
    if classification_errors and winner_count == unique_hash_count:
        print("Affected hashes use averages of only their successful estimates.")

    if not calculation_succeeded or not winner_count_matches:
        print("No files were deleted; resolution was not unambiguous.")
        for hash_value in ambiguous_hashes[:20]:
            print(f"Tied closest ages for hash {hash_value}")
        if len(ambiguous_hashes) > 20:
            print("Only the first 20 ambiguous hashes are shown.")
        return

    confirmation = input(
        "Delete every non-winning image listed in the CSV? Type 'yes' to "
        "continue: "
    )
    if confirmation.strip().casefold() != "yes":
        print("Cancelled; no files were deleted.")
        return

    deleted_count = 0
    deletion_errors = []
    for result in winners.values():
        for path in result["losers"]:
            try:
                Path(path).unlink()
            except OSError as exc:
                deletion_errors.append((path, str(exc)))
            else:
                deleted_count += 1
    print(f"Deleted {deleted_count} non-winning images.")
    print(f"Deletion failures: {len(deletion_errors)}")
    for path, error in deletion_errors[:20]:
        print(f"Could not delete {path}: {error}")
    if len(deletion_errors) > 20:
        print("Only the first 20 deletion failures are shown.")


if __name__ == "__main__":
    main()
