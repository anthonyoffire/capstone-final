import pandas as pd

#--------------------------------------------------------------------
# Find the number of fake files needed to achieve optimal split
# author: Anthony Baird
# algorithm: Craig Price and Kaden Davis
#--------------------------------------------------------------------

# CSV from predictor/csv_builder run on your real-person training set
CSV_PATH = "synthetic_csv/all.csv"

# Percentage of fakes desired
FAKE_PERCENT = 30.0
AGE_MIN = 5
AGE_MAX = 70

def main():
    df = pd.read_csv(CSV_PATH)
    real_counts = (
        df.groupby("age")
        .size()
        .reindex(range(AGE_MIN, AGE_MAX + 1), fill_value=0)
    )

    target_real_fraction = 1.0 - (FAKE_PERCENT / 100.0)
    best_bucket_size = None
    best_error = None
    max_bucket_size = int(real_counts.max())

    for bucket_size in range(max_bucket_size, -1, -1):
        total_size = bucket_size * len(real_counts)
        if total_size == 0:
            continue
        real_used = sum(
            min(count, bucket_size)
            for count in real_counts
        )
        actual_real_fraction = real_used / total_size
        error = abs(actual_real_fraction - target_real_fraction)
        if (
            best_error is None
            or error < best_error
            or (
                abs(error - best_error) < 1e-12
                and bucket_size > best_bucket_size
            )
        ):
            best_bucket_size = bucket_size
            best_error = error
        if error == 0:
            break
    rows = []

    for age, real_count in real_counts.items():
        fake_needed = max(0, best_bucket_size - real_count)

        if fake_needed > 0:
            rows.append({
                "age": age,
                "fake_needed": fake_needed,
            })
    out_csv = f"fake_needed_{int(100-FAKE_PERCENT)}real"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Wrote {len(rows)} ages to {out_csv}")
    print(f"Uniform bucket size: {best_bucket_size:,}")
    print(f"Total fake needed: {sum(r['fake_needed'] for r in rows):,}")

if __name__ == "__main__":
    main()