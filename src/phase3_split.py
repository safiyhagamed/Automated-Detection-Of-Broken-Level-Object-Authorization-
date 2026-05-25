from pathlib import Path
import pandas as pd

INPUT = Path("data/interim/featured_endpoints_safe.csv")
GOLD = Path("data/processed/gold_positives.csv")
TRAIN_OUT = Path("data/interim/train.csv")
TEST_OUT = Path("data/interim/test.csv")

def weak_label(row):
    score = 0
    if row["feat_has_identifier"]: score += 1
    if row["feat_method_risk"] >= 2: score += 1
    if row["feat_path_depth"] > 2: score += 1
    if row["feat_sensitive_path"]: score += 1
    if row["feat_anomaly"] > 0.6: score += 1
    return 1 if score >= 3 else 0

def main():
    df = pd.read_csv(INPUT, low_memory=False)

    required_cols = ["source_file", "path", "method"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["source_filename"] = df["source_file"].apply(lambda x: Path(x).name)

    gold = pd.read_csv(GOLD)
    gold_set = set(zip(gold["source_filename"], gold["path"].str.strip(), gold["method"].str.strip().str.upper()))

    df["is_gold_positive"] = df.apply(
        lambda r: (r["source_filename"], r["path"].strip(), r["method"].strip().upper()) in gold_set, axis=1
    )

    pos = df[df["is_gold_positive"]].copy()
    pos["label"] = 1
    remaining = df[~df["is_gold_positive"]].copy()

    # ---------- build TEST set ----------
    if len(remaining) < len(pos):
        raise ValueError("Not enough negative samples to balance test set")

    neg = remaining.sample(n=len(pos), random_state=42)
    neg["label"] = 0

    test_df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------- build TRAIN set ----------
    # FIX 1: Use test_original_idx to drop properly
    test_original_idx = pos.index.union(neg.index)
    train_df = df[~df.index.isin(test_original_idx)].copy()
    
    train_df["label"] = train_df.apply(weak_label, axis=1)

    # ---------- features ----------
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    cols = ["source_file"] + feat_cols + ["label"]

    # ---------- save ----------
    train_df[cols].to_csv(TRAIN_OUT, index=False)
    test_df[cols].to_csv(TEST_OUT, index=False)
    print("Phase 3 complete (fixed index bug)")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    main()