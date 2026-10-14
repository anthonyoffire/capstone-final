from pathlib import Path

from classify import classify_images


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "../images/real/uniform_split/all"
MODEL_TYPE = "densenet121"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / f"classifier_{MODEL_TYPE}.pt"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "classified" / MODEL_TYPE / "real" / Path(SOURCE_DIR).name


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {SOURCE_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {MODEL_PATH}")

    classify_images(SOURCE_DIR, OUTPUT_DIR, str(MODEL_PATH), "bad", "good")
    print(f"Classified images written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()