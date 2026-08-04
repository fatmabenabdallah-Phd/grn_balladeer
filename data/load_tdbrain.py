"""
grn_balladeer.data.load_tdbrain
==================================
Loads and filters the TDBRAIN participants metadata
(`TDBRAIN_participants_V3.xlsx`) into a clean, usable
ADHD-vs-Healthy label table.

Three real, verified issues in the raw participants file must be
controlled before any group comparison:

1. `indication` vs `formal_status`: the loose `indication` column
   over-counts "ADHD" subjects (200) relative to those with a
   confirmed `formal_status` of ADHD/ADD (104) -- the rest are
   `formal_status == "UNKNOWN"`. Always filter on `formal_status`.
2. Severe age mismatch between confirmed ADHD/ADD (6.0-55.0, includes
   children) and Healthy (18.0-82.7, adults only) groups. Any
   comparison on the unrestricted samples risks reflecting
   age-related EEG differences (maturation/aging), not ADHD itself.
   Fixed by restricting both groups to age >= min_age.
3. 4 duplicate `TDBRAIN_ID` rows (repeated clinical assessments, e.g.
   pre/post neurofeedback treatment, referencing the same single EEG
   session) -- deduplicated on `TDBRAIN_ID`, keeping the first row.
"""

from __future__ import annotations

import pandas as pd


def build_tdbrain_label_df(participants_xlsx_path: str, min_age: float = 18.0) -> pd.DataFrame:
    """Builds a clean [user_id, label, age, gender] label table from
    the raw TDBRAIN participants Excel file.

    label: 1 = ADHD/ADD (confirmed formal_status), 0 = HEALTHY.
    Prints before/after counts at each filtering step.

    user_id matches TDBRAIN_ID exactly as it appears in the
    participants file (e.g. "sub-88053677"), for direct use with
    find_tdbrain_subject_file.
    """
    df = pd.read_excel(participants_xlsx_path)
    n_total = len(df)
    print(f"[build_tdbrain_label_df] Lignes totales dans le fichier: {n_total}")

    # Step 1: filter on formal_status, not the loose 'indication' column
    df_confirmed = df[df["formal_status"].isin(["ADHD", "ADD", "HEALTHY"])].copy()
    n_adhd_confirmed = (df_confirmed["formal_status"].isin(["ADHD", "ADD"])).sum()
    n_healthy_confirmed = (df_confirmed["formal_status"] == "HEALTHY").sum()
    print(f"[build_tdbrain_label_df] Apres filtre formal_status confirme: "
          f"{n_adhd_confirmed} ADHD/ADD, {n_healthy_confirmed} HEALTHY "
          f"(total {len(df_confirmed)})")

    df_confirmed["label"] = df_confirmed["formal_status"].isin(["ADHD", "ADD"]).astype(int)

    # Step 2: age restriction, applied to BOTH groups identically
    df_age_ok = df_confirmed[df_confirmed["age"] >= min_age].copy()
    n_adhd_age = (df_age_ok["label"] == 1).sum()
    n_healthy_age = (df_age_ok["label"] == 0).sum()
    print(f"[build_tdbrain_label_df] Apres restriction age >= {min_age}: "
          f"{n_adhd_age} ADHD/ADD, {n_healthy_age} HEALTHY "
          f"(total {len(df_age_ok)})")

    # Step 3: deduplicate on TDBRAIN_ID, keep first occurrence
    n_before_dedup = len(df_age_ok)
    df_dedup = df_age_ok.drop_duplicates(subset="TDBRAIN_ID", keep="first").copy()
    n_after_dedup = len(df_dedup)
    if n_before_dedup != n_after_dedup:
        print(f"[build_tdbrain_label_df] {n_before_dedup - n_after_dedup} doublon(s) "
              f"TDBRAIN_ID retire(s) (garde la premiere occurrence)")
    print(f"[build_tdbrain_label_df] Echantillon final: "
          f"{(df_dedup['label'] == 1).sum()} ADHD/ADD, "
          f"{(df_dedup['label'] == 0).sum()} HEALTHY "
          f"(total {len(df_dedup)})")

    label_df = df_dedup.rename(columns={"TDBRAIN_ID": "user_id", "gender": "sex"})[
        ["user_id", "label", "age", "sex"]
    ].reset_index(drop=True)

    # age_bin: required by training.cross_validation.stratified_subject_kfold
    # (stratifies jointly on label + sex + age_bin). BALLADEER's own
    # age_bin convention (data/labels.py) uses child-specific bins
    # (6-9, 10-12, 13-15, 16-18) that do not apply to TDBRAIN's adult-only
    # population (18.0-82.7 after the age >= 18 restriction above) --
    # decade-based bins used instead, appropriate for this age range.
    label_df["age_bin"] = pd.cut(
        label_df["age"],
        bins=[17, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "61+"],
    )
    if label_df["age_bin"].isna().any():
        raise ValueError(
            "build_tdbrain_label_df: age_bin has NaN values -- check that "
            "all ages fall within [17, 100), the bin edges used above."
        )

    return label_df[["user_id", "label", "sex", "age", "age_bin"]]
