from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT_CSV = Path("data/interim/parsed_endpoints.csv")
OUTPUT_CSV = Path("data/interim/featured_endpoints_safe.csv")
RANDOM_STATE = 42
TRAIN_IDX_PATH = Path("data/interim/train_indices.csv") 

def path_depth(path):
    return len([s for s in path.strip().split("/") if s])

def is_admin(path):
    return 1 if "admin" in path.lower() else 0

def has_identifier(path):
    return 1 if re.search(r"\{.*?\}", path) else 0

def count_identifiers(path):
    return len(re.findall(r"\{.*?\}", path))

def method_risk(method):
    m = method.upper()
    if m == "GET": return 1
    if m == "POST": return 2
    if m in ["PUT", "PATCH", "DELETE"]: return 3
    return 1

def response_401(resp):
    if not isinstance(resp, str): return 0
    return 1 if "401" in resp or "403" in resp else 0

def sensitive_path(path):
    keywords = ["user", "account", "payment", "transaction", "profile"]
    return 1 if any(k in path.lower() for k in keywords) else 0

def id_position(path):
    parts = [p for p in path.strip().split("/") if p]
    for i, p in enumerate(parts):
        if "{" in p:
            return i / len(parts)
    return 0

def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # -------- base features --------
    df["feat_requires_auth"] = df["requires_auth"].astype(int)
    df["feat_path_depth"] = df["path"].apply(path_depth)
    df["feat_is_admin"] = df["path"].apply(is_admin)
    df["feat_has_identifier"] = df["path"].apply(has_identifier)
    df["feat_identifier_count"] = df["path"].apply(count_identifiers)
    df["feat_method_risk"] = df["method"].apply(method_risk)
    df["feat_param_count"] = np.log1p(df["total_parameters"].fillna(0))
    df["feat_response_401"] = df["response_codes"].apply(response_401)
    df["feat_sensitive_path"] = df["path"].apply(sensitive_path)
    df["feat_id_position"] = df["path"].apply(id_position)

    # -------- structural --------
    # REMOVED feat_is_nested and feat_id_and_depth as they were redundant
    df["feat_path_param_ratio"] = df["path_params"] / df["total_parameters"].replace(0, 1)

    # -------- interaction --------
    df["feat_id_and_write"] = df["feat_has_identifier"] * (df["feat_method_risk"] >= 2)
    df["feat_complex_risk"] = df["feat_has_identifier"] * df["feat_sensitive_path"] * df["feat_method_risk"]

    # -------- NEW BOLA FEATURES --------
    def param_name_id_risk(path):
        keywords = r"(user|account|order|customer)?id|uuid"
        params = re.findall(r"\{(.*?)\}", path.lower())
        return sum(1 for p in params if re.search(keywords, p))
    
    df["feat_param_name_id_risk"] = df["path"].apply(param_name_id_risk)
    df["feat_missing_auth_with_id"] = (df["feat_has_identifier"] == 1) & (df["feat_requires_auth"] == 0)
    df["feat_missing_auth_with_id"] = df["feat_missing_auth_with_id"].astype(int)

    # Sibling endpoint count
    resource_paths = df["path"].str.replace(r"/\{.*?\}$", "", regex=True)
    df["resource_path"] = resource_paths
    sibling_counts = df.groupby(["source_file", "resource_path"]).size().reset_index(name="sibling_method_count")
    df = df.merge(sibling_counts, on=["source_file", "resource_path"], how="left")
    df["feat_sibling_endpoint_count"] = df["sibling_method_count"] - 1 

    # -------- anomaly --------
    base_feats = [
        "feat_requires_auth", "feat_path_depth", "feat_is_admin",
        "feat_has_identifier", "feat_identifier_count", "feat_method_risk",
        "feat_param_count", "feat_response_401", "feat_sensitive_path",
        "feat_path_param_ratio"
    ]

    scaler = StandardScaler()
    apis_mask = df["source_file"].str.contains("apis_guru_flat", case=False, na=False)
    
    if apis_mask.sum() > 0:
        X_train = scaler.fit_transform(df[apis_mask][base_feats])
        X_all = scaler.transform(df[base_feats])
        
        iso = IsolationForest(
            n_estimators=200, contamination=0.05, 
            random_state=RANDOM_STATE, n_jobs=-1
        )
        iso.fit(X_train)
        scores = -iso.decision_function(X_all)
        
        # FIX 3: Normalize anomaly using bounds from train samples only
        try:
            train_idx_df = pd.read_csv(TRAIN_IDX_PATH)
            train_mask = df.index.isin(train_idx_df['index'])
            train_scores = scores[train_mask]
            
            df["feat_anomaly"] = (scores - train_scores.min()) / (train_scores.max() - train_scores.min())
        except FileNotFoundError:
            # Note: You should save train_indices.csv during train/test split to prevent leakage here
            df["feat_anomaly"] = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        df["feat_anomaly"] = 0.0

    cols = ["source_file", "path", "method"] + [c for c in df.columns if c.startswith("feat_")]
    df[cols].to_csv(OUTPUT_CSV, index=False)
    print("Phase 2 complete (safe features + BOLA additions)")

if __name__ == "__main__":
    main()