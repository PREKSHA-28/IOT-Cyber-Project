from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Dataset files
# ---------------------------------------------------------

FILES = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv",
]


# ---------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------

def inspect_file(file_path: Path) -> None:
    """Inspect one CIC-IoT-2023 dataset file."""

    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    print(f"Rows            : {len(df):,}")
    print(f"Columns         : {len(df.columns):,}")
    print(f"Missing values  : {df.isna().sum().sum():,}")
    print(f"Duplicate rows  : {df.duplicated().sum():,}")

    print("\nColumn names:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nLabel distribution:")
    print(df["Label"].value_counts(dropna=False))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Inspect all dataset files."""

    for filename in FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"\nERROR: File not found -> {file_path}")
            continue

        inspect_file(file_path)


if __name__ == "__main__":
    main()