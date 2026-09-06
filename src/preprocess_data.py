from pathlib import Path
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Input files
# ============================================================

DATA_FILES = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv",
]


# ============================================================
# Preprocessing
# ============================================================

def preprocess_file(file_path: Path) -> pd.DataFrame:
    """
    Load and clean one CIC-IoT-2023 file.

    Returns:
        Cleaned DataFrame with:
        - numeric feature columns
        - original multiclass Label
        - binary Label_Binary
    """

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path)

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates()

    duplicates_removed = before_duplicates - len(df)

    print(f"Duplicates removed: {duplicates_removed:,}")

    # --------------------------------------------------------
    # Remove rows with missing target labels
    # --------------------------------------------------------

    before_missing_label = len(df)

    df = df.dropna(subset=["Label"])

    missing_labels_removed = before_missing_label - len(df)

    print(f"Rows with missing labels removed: {missing_labels_removed:,}")

    # --------------------------------------------------------
    # Clean label text
    # --------------------------------------------------------

    df["Label"] = df["Label"].astype(str).str.strip()

    # --------------------------------------------------------
    # Create binary target
    #
    # BENIGN  -> BENIGN
    # Everything else -> ATTACK
    # --------------------------------------------------------

    df["Label_Binary"] = df["Label"].apply(
        lambda label: "BENIGN" if label == "BENIGN" else "ATTACK"
    )

    # --------------------------------------------------------
    # Replace infinite numeric values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # --------------------------------------------------------
    # Remove rows containing invalid numeric values
    # --------------------------------------------------------

    before_invalid = len(df)

    df = df.dropna(subset=numeric_columns)

    invalid_rows_removed = before_invalid - len(df)

    print(f"Rows with invalid numeric values removed: {invalid_rows_removed:,}")

    print(f"Final rows: {len(df):,}")

    return df


# ============================================================
# Main
# ============================================================

def main() -> None:

    # Create processed-data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_frames = []

    for filename in DATA_FILES:

        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            print(f"\nERROR: File not found: {file_path}")
            continue

        df = preprocess_file(file_path)

        processed_frames.append(df)

    # --------------------------------------------------------
    # Make sure files were loaded
    # --------------------------------------------------------

    if not processed_frames:
        print("\nNo dataset files were processed.")
        return

    # --------------------------------------------------------
    # Combine all five parts
    # --------------------------------------------------------

    print("\nCombining dataset parts...")

    combined_df = pd.concat(
        processed_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final duplicate check
    # --------------------------------------------------------

    before_final_duplicates = len(combined_df)

    combined_df = combined_df.drop_duplicates()

    final_duplicates_removed = (
        before_final_duplicates - len(combined_df)
    )

    print(
        f"Duplicates removed after combining: "
        f"{final_duplicates_removed:,}"
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DATA_DIR /
        "ciciot2023_processed.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"Final rows    : {len(combined_df):,}")
    print(f"Final columns : {len(combined_df.columns):,}")

    print("\nBinary label distribution:")
    print(combined_df["Label_Binary"].value_counts())

    print("\nMulticlass label distribution:")
    print(combined_df["Label"].value_counts())

    print(f"\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()