# build_features.py
from pathlib import Path
import pandas as pd

from preprocessing.cleaner import process_text_features
from preprocessing.dataset_builder import build_dataset

IN_CLEAN_CSV = Path("data/processed/social_clean.csv")
OUT_CSV = Path("data/processed/social_wide_final.csv")

def add_is_viral_by_er_final(df: pd.DataFrame) -> pd.DataFrame:
    """is_viral = top 10% by er_final within platform."""
    if "er_final" not in df.columns:
        print("[warn] 'er_final' not found — skip is_viral.")
        return df
    if "platform" in df.columns:
        thr = df.groupby(df["platform"].astype(str).str.lower())["er_final"] \
                .transform(lambda s: s.quantile(0.90))
    else:
        thr = pd.Series(df["er_final"].quantile(0.90), index=df.index)
    df["is_viral"] = (df["er_final"] >= thr).astype(int)
    return df

def main():
    # Step 1 (optional): clean raw -> social_clean.csv
    print("[1/3] Loading raw CSV ...")
    df_raw = pd.read_csv("data/raw/social_wide.csv")
    print(f"      rows={len(df_raw):,}, cols={df_raw.shape[1]}")
    print("[2/3] Cleaning text ...")
    df_clean = process_text_features(df_raw)
    IN_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(IN_CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"      saved -> {IN_CLEAN_CSV}")

    # Step 2: build features on top of cleaned CSV
    print("[1/3] Loading cleaned CSV ...")
    if not IN_CLEAN_CSV.exists():
        raise FileNotFoundError(f"{IN_CLEAN_CSV} not found")
    df_clean = pd.read_csv(IN_CLEAN_CSV)
    print(f"      rows={len(df_clean):,}, cols={df_clean.shape[1]}")

    print("[2/3] Building features ...")
    before_cols = set(df_clean.columns)
    df_final = build_dataset(
        df_clean,
        compute_viral_label=False,        
        enable_sentiment_if_missing=False  
    )
    added = sorted(list(set(df_final.columns) - before_cols))
    print(f"      done. +{len(added)} new cols: {', '.join(added[:10])}{' ...' if len(added)>10 else ''}")

    print("[3/3] Computing target is_viral (top 10% by er_final per platform) ...")
    df_final = add_is_viral_by_er_final(df_final)
    if "platform" in df_final.columns and "is_viral" in df_final.columns:
        share = df_final.groupby(df_final["platform"].astype(str))["is_viral"].mean().round(3).to_dict()
        print(f"      is_viral share by platform: {share}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✓ saved: {OUT_CSV} | rows={len(df_final):,}, cols={df_final.shape[1]}")

if __name__ == "__main__":
    main()

