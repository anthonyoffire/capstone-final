import os
import pandas as pd

aggregation_directories = [
    "dataset/generation/gen/images/serenity", 
]
save_names = []
minAge = 5
maxAge = 70
def main():
    for index, aggregation_directory in enumerate(aggregation_directories):
        listRows = []

        for folder in os.listdir(aggregation_directory):
            path = os.path.join(aggregation_directory, folder)
            age = int(folder)

            files = []
            for file in os.listdir(path):
                files.append(os.path.join(path, file))

            for file_path in files:
                listRows.append({
                    "file_path": file_path,
                    "age": age
                })

        df = pd.DataFrame(listRows)

        df.to_csv("./synthetic_csv/" + save_names[index], index=False)

        print(save_names[index])
        print(df.head())
        print(len(df))
        print(min(df["age"]))
        print(max(df["age"]))

if __name__ == "__main__":
    for dir in aggregation_directories:
        save_names.append(os.path.basename(dir.rstrip("/\\")) + ".csv")
    main()