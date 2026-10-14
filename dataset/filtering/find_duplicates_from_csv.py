import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
from pathlib import Path

# Global variables for input and output CSV file paths
INPUT_CSV = "eval_outputs_efficientnet_b4_g1.pth_all-minus-noface/age_predictions.csv"
OUTPUT_CSV = "dataset/filtering/duplicates.csv"
OUT_DIR = "dataset/filtering"
COPY = True

def hash_file(file_path):
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in chunks to handle large files
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None


def process_file(file_path):
    """Process a single file: compute its hash and return (hash, file_path)."""
    file_hash = hash_file(file_path)
    if file_hash is not None:
        return file_hash, file_path
    return None, None


def find_duplicates_from_csv(csv_file):
    """Read CSV of file paths, hash files in parallel, and find duplicates."""
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    if 'file_path' not in df.columns:
        raise ValueError("CSV must contain 'file_path' column.")
    
    file_paths = df['file_path'].tolist()
    
    # Dictionary to store hashes and their corresponding file paths
    hash_to_files = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, path): path for path in file_paths}
        
        # Process completed tasks with a progress bar
        with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
            for future in as_completed(futures):
                hash_val, file_path = future.result()
                if hash_val is not None and file_path is not None:
                    if hash_val not in hash_to_files:
                        hash_to_files[hash_val] = []
                    hash_to_files[hash_val].append(file_path)
                pbar.update(1)
    
    # Filter out non-duplicates (hashes with only one file)
    duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}
    
    return df, duplicates


def save_duplicates(df_original, duplicates):
    """Save duplicate files with hash, file_path, predicted_age."""

    age_lookup = dict(
        zip(df_original["file_path"], df_original["detected_age"])
    )
    rows = []
    for file_hash, paths in duplicates.items():
        for path in paths:
            rows.append({
                "hash": file_hash,
                "file_path": path,
                "predicted_age": age_lookup.get(path)
            })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(rows)} duplicate file entries to: {OUTPUT_CSV}")

def get_avg_label_range(duplicates):
    age_ranges = []

    for paths in duplicates.values():
        ages = []

        for path in paths:
            try:
                ages.append(int(Path(path).parent.name))
            except ValueError:
                continue

        if ages:
            age_ranges.append(max(ages) - min(ages))

    avg_age_range = (
        sum(age_ranges) / len(age_ranges)
        if age_ranges else 0
    )
    return avg_age_range

def main():
    """Main function to find and save duplicate files from CSV."""
    print(f"Reading CSV: {INPUT_CSV}")
    
    # Load the dataset
    df_original, duplicates = find_duplicates_from_csv(INPUT_CSV)
    
    duplicate_hashes = len(duplicates)

    duplicate_files = sum(len(paths) for paths in duplicates.values())

    extra_duplicate_copies = sum(len(paths) - 1 for paths in duplicates.values())

    print(f"Total number of images: {len(df_original)}")
    print(f"Duplicate groups: {duplicate_hashes}")
    print(f"Images participating in duplicates: {duplicate_files}")
    print(f"Extra duplicate copies: {extra_duplicate_copies}")
    print(f"Average duplicate label range: {get_avg_label_range(duplicates)}")
    if duplicates:
        print("Saving duplicates and predicted ages to CSV...")
        save_duplicates(df_original, duplicates)
    else:
        print("No duplicates found.")


if __name__ == "__main__":
    main()
