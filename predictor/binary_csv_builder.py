import argparse
import csv
import math
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Use dist type "cn" for train and val, "all" for test
#   - Age cutoff only affects cn
#   - Use 1 train and 1 val csv per target age
#   - Use 1 test csv for all target ages
#----------------------------------------------------------------

INPUT_DIR = Path("../images/ser8020_bin/test")
OUTPUT_DIR = Path("./synthetic_csv")
DISTRIBUTION_GRAPH_DIR = OUTPUT_DIR / "distribution_graphs"
MAKE_GRAPH = True

DISTRIBUTION_TYPE = "all"
AGE_CUTOFF = 12

RANDOM_SEED = 42
MIN_AGE = 5
MAX_AGE = 70

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DISTRIBUTIONS = {
    "all": "All Files",
    "cn": "Centered Normal (balance_app's mean x)",
}
CSV_FIELDS = (
    "file_path",
    "age",
    "binary_label", # Only written if 'cn', not 'all'
)

CENTERED_NORMAL_BELOW_AMPLITUDE = 0.064238128651353
CENTERED_NORMAL_BELOW_SIGMA = 6.50641037886466
CENTERED_NORMAL_ABOVE_SIGMA = 26.5015723284053

def _largest_remainder_allocation(
    total_size: int,
    probabilities: list[float],
    capacities: list[int],
) -> list[int] | None:
    if total_size < 0 or total_size > sum(capacities):
        return None

    raw = [probability * total_size for probability in probabilities]
    allocation = [math.floor(value) for value in raw]
    if any(count > capacity for count, capacity in zip(allocation, capacities)):
        return None

    remainder = total_size - sum(allocation)
    remainders = sorted(
        [
            (raw[index] - allocation[index],index,)
            for index in range(len(probabilities))
        ],
        reverse=True,
    )

    for _, index in remainders:
        if remainder == 0:
            break
        if allocation[index] < capacities[index]:
            allocation[index] += 1
            remainder -= 1

    return allocation if remainder == 0 else None


def _normalize_weights(weights: list[float], description: str) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"No available probability mass for {description}.")
    return [weight / total for weight in weights]


def _scan_input(input_root: Path) -> dict[int, list[Path]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    files_by_age = {}
    with os.scandir(input_root) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.isdecimal():
                age = int(entry.name)
                if age in files_by_age:
                    raise ValueError(
                        f"Multiple input folders represent age {age}: "
                        f"{entry.name!r}."
                    )
                files = []
                with os.scandir(entry.path) as age_entries:
                    for age_entry in age_entries:
                        if (
                            age_entry.is_file()
                            and Path(age_entry.name).suffix.lower()
                            in IMAGE_EXTENSIONS
                        ):
                            files.append(Path(age_entry.path))
                files.sort()
                files_by_age[age] = files

    return dict(sorted(files_by_age.items()))


def _centered_normal_weights() -> list[float]:
    center = AGE_CUTOFF - 0.5
    below_ages = range(MIN_AGE, AGE_CUTOFF)
    at_or_above_ages = range(AGE_CUTOFF, MAX_AGE + 1)

    below_shape_total = sum(
        math.exp(
            -((age - center) ** 2)
            / (2 * CENTERED_NORMAL_BELOW_SIGMA**2)
        )
        for age in below_ages
    )
    at_or_above_shape_total = sum(
        math.exp(
            -((age - center) ** 2)
            / (2 * CENTERED_NORMAL_ABOVE_SIGMA**2)
        )
        for age in at_or_above_ages
    )
    if below_shape_total == 0 or at_or_above_shape_total == 0:
        raise ValueError(
            "Centered-normal distribution cutoff must leave ages on both sides "
            f"within {MIN_AGE}..{MAX_AGE}."
        )

    above_amplitude = (
        CENTERED_NORMAL_BELOW_AMPLITUDE
        * below_shape_total
        / at_or_above_shape_total
    )
    weights = [0.0] * (MAX_AGE + 1)
    for age in range(MIN_AGE, MAX_AGE + 1):
        if age < AGE_CUTOFF:
            amplitude = CENTERED_NORMAL_BELOW_AMPLITUDE
            sigma = CENTERED_NORMAL_BELOW_SIGMA
        else:
            amplitude = above_amplitude
            sigma = CENTERED_NORMAL_ABOVE_SIGMA
        weights[age] = amplitude * math.exp(
            -((age - center) ** 2) / (2 * sigma**2)
        )
    return weights


def _allocate_centered_normal(
    ages: list[int],
    total_counts: dict[int, int],
) -> dict[int, int]:
    weights = _centered_normal_weights()
    probabilities = _normalize_weights(
        [max(0.0, float(weights[age])) for age in ages],
        "centered-normal formula weights",
    )
    capacities = [total_counts[age] for age in ages]
    for total_size in range(sum(capacities), 0, -1):
        allocation = _largest_remainder_allocation(
            total_size,
            probabilities,
            capacities,
        )
        if allocation is not None:
            return dict(zip(ages, allocation))
    raise ValueError(
        f"Could not allocate centered_normal at cutoff {AGE_CUTOFF}."
    )


def _iter_csv_rows(
    files_by_age: dict[int, list[Path]],
    bucket_sizes: dict[int, int],
):
    rng = random.Random(RANDOM_SEED)
    selected_by_age: dict[int, list[Path]] = {}

    for age in sorted(files_by_age):
        pool = list(files_by_age[age])
        rng.shuffle(pool)
        selected_by_age[age] = sorted(pool[:bucket_sizes[age]])

    for age in sorted(selected_by_age, key=str):
        for path in selected_by_age[age]:
            row = {
                "file_path": str(path),
                "age": age,
            }
            if DISTRIBUTION_TYPE != "all":
                row["binary_label"] = int(age >= AGE_CUTOFF)
            yield row


def write_distribution_csv(
    output_path: Path,
    rows,
) -> int:
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = CSV_FIELDS[:2] if DISTRIBUTION_TYPE == "all" else CSV_FIELDS
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def save_distribution_graph(
    csv_path: Path,
    ages: list[int],
    age_counts: dict[int, int],
) -> Path:
    DISTRIBUTION_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = DISTRIBUTION_GRAPH_DIR / f"{csv_path.stem}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ages, [age_counts[age] for age in ages], color="steelblue")
    if DISTRIBUTION_TYPE != "all":
        ax.axvline(
            AGE_CUTOFF - 0.5,
            color="crimson",
            linestyle="--",
            label=f"Age cutoff: {AGE_CUTOFF}",
        )
    ax.set(
        title=f"{DISTRIBUTIONS[DISTRIBUTION_TYPE]} age distribution",
        xlabel="Age",
        ylabel="Number of images",
        xticks=ages,
    )
    if DISTRIBUTION_TYPE != "all":
        ax.legend()
    fig.tight_layout()
    fig.savefig(graph_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution graph: {graph_path}")
    return graph_path


def build_binary_csv() -> Path:
    if DISTRIBUTION_TYPE not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution type {DISTRIBUTION_TYPE!r}. "
            f"Choose from: {', '.join(DISTRIBUTIONS)}."
        )
    print(f"Age cutoff: {AGE_CUTOFF}")
    files_by_age = {
        age: files
        for age, files in _scan_input(INPUT_DIR.resolve()).items()
        if MIN_AGE <= age <= MAX_AGE
    }
    if not files_by_age:
        raise ValueError(
            f"Input directory contains no numeric age folders in {MIN_AGE}..{MAX_AGE}."
        )
    ages = list(range(MIN_AGE, MAX_AGE + 1))
    for age in ages:
        files_by_age.setdefault(age, [])
    total_counts = {age: len(files_by_age[age]) for age in ages}

    if not any(age < AGE_CUTOFF for age in ages) or not any(
        age >= AGE_CUTOFF for age in ages
    ):
        raise ValueError(
            "The age cutoff must leave age folders both below and at/above it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_str = f'{AGE_CUTOFF if DISTRIBUTION_TYPE != "all" else ""}'
    output_path = OUTPUT_DIR / (
        f"{INPUT_DIR.name}_{DISTRIBUTION_TYPE}{cutoff_str}.csv"
    )
    bucket_sizes = (
        total_counts
        if DISTRIBUTION_TYPE == "all"
        else _allocate_centered_normal(ages, total_counts)
    )

    row_count = write_distribution_csv(
        output_path,
        _iter_csv_rows(files_by_age, bucket_sizes),
    )
    if MAKE_GRAPH:
        save_distribution_graph(output_path, ages, bucket_sizes)
    print(
        f"{DISTRIBUTIONS[DISTRIBUTION_TYPE]}: "
        f"wrote {row_count:,} rows to {output_path}"
    )
    return output_path

def main() -> None:
    global AGE_CUTOFF, DISTRIBUTION_GRAPH_DIR, DISTRIBUTION_TYPE
    global INPUT_DIR, MAKE_GRAPH, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Build a binary-classification CSV from age-organized image folders."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing one subdirectory per age.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the CSV will be written.",
    )
    parser.add_argument(
        "-d",
        "--distribution-type",
        choices=DISTRIBUTIONS,
        default=DISTRIBUTION_TYPE,
        help="Use 'cn' for centered-normal train/validation data or 'all' for test data.",
    )
    parser.add_argument(
        "-a",
        "--age-cutoff",
        type=int,
        default=AGE_CUTOFF,
        help="Age boundary used for the centered-normal distribution and binary labels.",
    )
    parser.add_argument(
        "-n",
        "--no-graph",
        action="store_false",
        dest="make_graph",
        default=MAKE_GRAPH,
        help="Do not save an age-distribution graph.",
    )
    args = parser.parse_args()

    AGE_CUTOFF = args.age_cutoff
    DISTRIBUTION_TYPE = args.distribution_type
    INPUT_DIR = args.input_dir
    MAKE_GRAPH = args.make_graph
    OUTPUT_DIR = args.output_dir
    DISTRIBUTION_GRAPH_DIR = OUTPUT_DIR / "distribution_graphs"
    build_binary_csv()


if __name__ == "__main__":
    main()
