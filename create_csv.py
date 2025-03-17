import sys
import pandas as pd

for split in ['train', 'test']:
    # Load the TSV file
    df = pd.read_csv(f"data/{split}.tsv", sep="\t")
    # If test split, take only the first 362 subjects
    # This dataset gave the labels but not the inputs for another ~400 subjects
    if split == "test":
        df = df.iloc[:362]
    # Create a new sequential participant ID column
    df.insert(0, "t1_path", [f"{split}/{i:04d}_T1.npy" for i in range(len(df))])
    # Keep only the required columns
    df = df[["t1_path", "gmv"]]
    # Save to CSV
    df.to_csv(f"{split}.csv", index=False, sep=",")

print("Done!")
