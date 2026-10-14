from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from tqdm import tqdm


INPUT_DIR = Path("../images/fake/sam")
OUTPUT_DIR = Path("dataset/no_dups") / INPUT_DIR.name


def hash_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def detect_unique_files(input_dir: Path, output_dir: Path) -> tuple[list[Path], int, int]:
    first_path_by_hash: dict[str, Path] = {}
    duplicate_count = 0
    output_dir = output_dir.resolve()
    paths = [
        path for path in sorted(input_dir.rglob("*"))
        if path.is_file()
        and output_dir not in path.resolve().parents
    ]
    unique_paths = []
    files_hashed = 0

    with tqdm(total=len(paths), desc="Detecting duplicates", unit="file") as progress:
        for path in paths:
            try:
                digest = hash_file(path)
                files_hashed += 1
                if digest in first_path_by_hash:
                    duplicate_count += 1
                else:
                    first_path_by_hash[digest] = path
                    unique_paths.append(path)
            except OSError as error:
                tqdm.write(f"Could not hash {path}: {error}")
            finally:
                progress.update(1)

    return unique_paths, duplicate_count, files_hashed


def copy_unique_files(input_dir: Path, output_dir: Path, unique_paths: list[Path]) -> None:
    output_dir = output_dir.resolve()
    with tqdm(total=len(unique_paths), desc="Copying unique files", unit="file") as progress:
        for path in unique_paths:
            try:
                relative_path = path.relative_to(input_dir)
                destination = output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            except OSError as error:
                tqdm.write(f"Could not copy {path}: {error}")
            finally:
                progress.update(1)


def main() -> None:
    input_dir = INPUT_DIR.expanduser()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    output_dir = OUTPUT_DIR.expanduser()
    unique_paths, duplicate_count, files_hashed = detect_unique_files(input_dir, output_dir)
    print(f"{len(unique_paths)}/{files_hashed} files are unique. {duplicate_count} are duplicates.")

    try:
        response = input("Copy unique files? [y/N]: ").strip().lower()
    except EOFError:
        response = ""
    if response in {"y", "yes"}:
        copy_unique_files(input_dir, output_dir, unique_paths)
    else:
        print("Copy cancelled; no files were copied.")


if __name__ == "__main__":
    main()
