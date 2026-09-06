from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ciciot2023_processed.csv"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    if not PROCESSED_FILE.exists():
        print(f"ERROR: File not found: {PROCESSED_FILE}")
        return

    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_FILE)

    print("\n" + "=" * 70)
    print("PROCESSED DATASET SUMMARY")
    print("=" * 70)

    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns):,}")
    print(f"Missing values    : {df.isna().sum().sum():,}")
    print(f"Duplicate rows    : {df.duplicated().sum():,}")

    print("\nBinary label distribution:")
    print(df["Label_Binary"].value_counts())

    print("\nBinary label percentages:")
    print(
        df["Label_Binary"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMulticlass labels:")
    print(f"Number of attack classes: {df['Label'].nunique() - 1}")

    print("\nFeature columns:")
    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    print(f"Number of features: {len(feature_columns)}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()