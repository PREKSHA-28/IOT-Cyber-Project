from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ciciot2023_processed.csv"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# Attack classes intentionally held out from training.
# These will be used as our emerging/unseen attack evaluation set.
EMERGING_ATTACKS = {
    "DNS_SPOOFING",
    "VULNERABILITYSCAN",
    "DOS-HTTP_FLOOD",
}


# ============================================================
# Helper functions
# ============================================================

def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary BENIGN/ATTACK target."""

    df = df.copy()

    df["Label_Binary"] = df["Label"].apply(
        lambda label: "BENIGN" if label == "BENIGN" else "ATTACK"
    )

    return df


def split_known_data(df: pd.DataFrame):
    """
    Split known-attack data into training, validation and known-test sets.

    The split is stratified using the multiclass attack label so that the
    known attack categories remain represented in all three partitions.
    """

    # First split: training vs temporary set
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["Label"],
    )

    # Second split: validation vs known test
    validation_df, known_test_df = train_test_split(
        temp_df,
        test_size=(1 / 1.5),
        random_state=RANDOM_STATE,
        stratify=temp_df["Label"],
    )

    return train_df, validation_df, known_test_df


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not PROCESSED_FILE.exists():
        print(f"ERROR: Processed dataset not found:")
        print(PROCESSED_FILE)
        return

    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_FILE)

    print(f"Total rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Ensure binary label exists
    # --------------------------------------------------------

    df = create_binary_label(df)

    # --------------------------------------------------------
    # Separate emerging attacks
    # --------------------------------------------------------

    emerging_mask = df["Label"].isin(EMERGING_ATTACKS)

    emerging_df = df[emerging_mask].copy()

    known_df = df[~emerging_mask].copy()

    print("\n" + "=" * 70)
    print("EMERGING ATTACK SEPARATION")
    print("=" * 70)

    print(f"Known-data rows     : {len(known_df):,}")
    print(f"Emerging-data rows  : {len(emerging_df):,}")

    print("\nEmerging attack classes:")
    print(emerging_df["Label"].value_counts())

    # --------------------------------------------------------
    # Split known data
    # --------------------------------------------------------

    print("\nSplitting known data...")

    train_df, validation_df, known_test_df = split_known_data(
        known_df
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert not set(train_df["Label"]).intersection(EMERGING_ATTACKS)
    assert not set(validation_df["Label"]).intersection(EMERGING_ATTACKS)
    assert not set(known_test_df["Label"]).intersection(EMERGING_ATTACKS)

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(f"Training rows       : {len(train_df):,}")
    print(f"Validation rows     : {len(validation_df):,}")
    print(f"Known test rows     : {len(known_test_df):,}")
    print(f"Emerging test rows  : {len(emerging_df):,}")

    # --------------------------------------------------------
    # Binary distributions
    # --------------------------------------------------------

    print("\nTraining binary distribution:")
    print(train_df["Label_Binary"].value_counts())

    print("\nValidation binary distribution:")
    print(validation_df["Label_Binary"].value_counts())

    print("\nKnown-test binary distribution:")
    print(known_test_df["Label_Binary"].value_counts())

    print("\nEmerging-test binary distribution:")
    print(emerging_df["Label_Binary"].value_counts())

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"
    known_test_path = SPLIT_DIR / "known_test.csv"
    emerging_test_path = SPLIT_DIR / "emerging_test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    known_test_df.to_csv(known_test_path, index=False)
    emerging_df.to_csv(emerging_test_path, index=False)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 70)

    print(f"Training data      : {train_path}")
    print(f"Validation data    : {validation_path}")
    print(f"Known test data    : {known_test_path}")
    print(f"Emerging test data : {emerging_test_path}")


if __name__ == "__main__":
    main()