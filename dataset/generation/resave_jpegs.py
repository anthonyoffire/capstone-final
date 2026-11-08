from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

SOURCE_DIR = r"../images/fake/serenity"
OUTPUT_DIR = r"../images/fake/serenity_85"
JPEG_QUALITY = 85

WORKERS = 16


def process_image(src_path: Path):
    rel_path = src_path.relative_to(SOURCE_DIR)
    dst_path = Path(OUTPUT_DIR) / rel_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        img.save(
            dst_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )


def main():
    source = Path(SOURCE_DIR)

    jpgs = [
        p
        for p in source.rglob("*.jpg")
    ]

    print(f"Found {len(jpgs):,} JPG files")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(
            tqdm(
                executor.map(process_image, jpgs),
                total=len(jpgs),
                unit="img",
            )
        )

    print("Done")


if __name__ == "__main__":
    main()