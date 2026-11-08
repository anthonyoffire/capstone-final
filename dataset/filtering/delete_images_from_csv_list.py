import os
import pandas as pd

CSV_PATH = "eval_outputs_efficientnet_b4_realonly.pth_images_under_60/age_predictions.csv"

df = pd.read_csv(CSV_PATH)

deleted = 0
missing = 0
errors = 0

for file_path in df["file_path"]:
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted += 1
        else:
            missing += 1
    except Exception as e:
        errors += 1
        print(f"ERROR: {file_path} -> {e}")

print(f"Deleted: {deleted}")
print(f"Missing: {missing}")
print(f"Errors: {errors}")