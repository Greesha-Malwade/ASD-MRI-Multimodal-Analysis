import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import shared_data_utils as sdu


def assign_severity(score):
    """
    Convert ADOS-2 severity score into the
    project's proposed 3-class severity labels.

    1–5   -> Low
    6–7   -> Moderate
    8–10  -> High
    """

    if 1 <= score <= 5:
        return "Low"

    elif 6 <= score <= 7:
        return "Moderate"

    elif 8 <= score <= 10:
        return "High"

    else:
        return None


def main():

    print("=" * 60)
    print("PERSON 2 - SEVERITY COHORT VERIFICATION")
    print("=" * 60)

    # ========================================================
    # 1. LOAD OFFICIAL TRAIN / VALIDATION / TEST SPLITS
    # ========================================================

    print("\n[1] Loading official splits...")

    train = sdu.load_split("train")
    val = sdu.load_split("val")
    test = sdu.load_split("test")

    # Add split identifier
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    # Combine all official splits
    split_df = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    # IMPORTANT:
    # Make participant_id the same datatype as the
    # phenotypic dataset before merging.
    split_df["participant_id"] = (
        split_df["participant_id"]
        .astype(str)
        .str.strip()
    )

    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Total: {len(split_df)}")

    # ========================================================
    # 2. LOAD PHENOTYPIC DATA
    # ========================================================

    print("\n[2] Loading ADOS-2 phenotypic data...")

    # Project root = mri 2
    project_root = Path(__file__).resolve().parents[1]

    phenotypic_path = (
        project_root
        / "data"
        / "phenotypic"
        / "abide2_phenotypic.csv"
    )

    phenotypic = pd.read_csv(phenotypic_path)

    print(f"Phenotypic rows: {len(phenotypic)}")

    # ========================================================
    # 3. SELECT ASD SUBJECTS WITH VALID ADOS-2
    # ========================================================

    print("\n[3] Selecting ASD subjects with valid ADOS-2...")

    severity = phenotypic[
        (phenotypic["dx_group"] == 1)
        &
        (phenotypic["ados_2_severity_total"].notna())
    ][
        [
            "participant_id",
            "ados_2_severity_total"
        ]
    ].copy()

    # IMPORTANT:
    # Make participant_id compatible with split_df.
    severity["participant_id"] = (
        severity["participant_id"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Valid ASD severity subjects: {len(severity)}"
    )

    # ========================================================
    # 4. ASSIGN PROPOSED 3-CLASS SEVERITY LABELS
    # ========================================================

    print("\n[4] Assigning working severity classes...")

    severity["severity_class"] = (
        severity["ados_2_severity_total"]
        .apply(assign_severity)
    )

    print("\nSeverity distribution:")

    severity_distribution = (
        severity["severity_class"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"],
            fill_value=0
        )
    )

    print(severity_distribution)

    # ========================================================
    # 5. MERGE WITH OFFICIAL SPLIT
    # ========================================================

    print("\n[5] Merging severity subjects with official split...")

    cohort = split_df.merge(
        severity,
        on="participant_id",
        how="inner"
    )

    print(
        f"Subjects after merge: {len(cohort)}"
    )

    # ========================================================
    # 6. VERIFY SPLIT DISTRIBUTION
    # ========================================================

    print("\n[6] Split distribution:")

    split_distribution = (
        cohort["split"]
        .value_counts()
        .reindex(
            ["train", "val", "test"],
            fill_value=0
        )
    )

    print(split_distribution)

    # ========================================================
    # 7. VERIFY SPLIT × SEVERITY
    # ========================================================

    print("\n[7] Split × Severity:")

    cross_tab = pd.crosstab(
        cohort["split"],
        cohort["severity_class"]
    )

    cross_tab = cross_tab.reindex(
        index=["train", "val", "test"],
        columns=["Low", "Moderate", "High"],
        fill_value=0
    )

    print(cross_tab)

    # ========================================================
    # 8. CHECK FUNCTIONAL MRI AVAILABILITY
    # ========================================================

    print("\n[8] Functional MRI availability:")

    if "has_func_available" in cohort.columns:

        func_counts = (
            cohort["has_func_available"]
            .value_counts(dropna=False)
        )

        print(func_counts)

        # Handle boolean / numeric representations safely
        func_available = (
            cohort["has_func_available"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        print(
            "\nAvailable functional MRI subjects:",
            int(func_available.sum())
        )

        print("\nFunctional MRI availability by split:")

        func_by_split = pd.crosstab(
            cohort["split"],
            func_available
        )

        print(func_by_split)

    else:

        print(
            "WARNING: has_func_available column "
            "was not found."
        )

    # ========================================================
    # 9. SHOW FIRST 10 COHORT SUBJECTS
    # ========================================================

    print("\n[9] First 10 cohort subjects:")

    display_columns = [
        "participant_id",
        "site_code",
        "split",
        "ados_2_severity_total",
        "severity_class",
        "has_func_available"
    ]

    # Only display columns that actually exist
    display_columns = [
        column
        for column in display_columns
        if column in cohort.columns
    ]

    print(
        cohort[display_columns]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # 10. SAVE PERSON 2 COHORT
    # ========================================================

    print("\n[10] Saving Person 2 cohort...")

    output_dir = (
        project_root
        / "data"
        / "phenotypic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "person2_severity_fmri_cohort.csv"
    )

    cohort.to_csv(
        output_path,
        index=False
    )

    print("\nSaved to:")
    print(output_path)

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 2 COHORT VERIFICATION COMPLETE")
    print("=" * 60)

    print("\nFinal cohort size:", len(cohort))

    print("\nFinal severity counts:")

    print(
        cohort["severity_class"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"],
            fill_value=0
        )
    )

    print("\nFinal split counts:")

    print(
        cohort["split"]
        .value_counts()
        .reindex(
            ["train", "val", "test"],
            fill_value=0
        )
    )


if __name__ == "__main__":
    main()