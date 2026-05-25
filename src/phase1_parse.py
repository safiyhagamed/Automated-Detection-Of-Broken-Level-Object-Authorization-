from pathlib import Path
import json
import yaml
import pandas as pd

GOLD_DIRS = [
    Path("data/raw/vampi"),
    Path("data/raw/crapi"),
    Path("data/raw/dvws"),
    Path("data/raw/juiceshop"),
    Path("data/raw/vapi"),
    Path("data/raw/vuln-bank"),
]
API_GURU_DIR = Path("data/raw/apisguru_flat")

VALID_SUFFIXES = {".json", ".yaml", ".yml"}
OUTPUT_PATH = Path("data/interim/parsed_endpoints.csv")

HTTP_METHODS = {"get","post","put","patch","delete","head","options"}

IGNORE_PATHS = ["health", "status", "ping"]

# ---------- helpers ----------

def load_spec(fp):
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if fp.suffix.lower() in {".yaml",".yml"}:
        return yaml.safe_load(text)
    if fp.suffix.lower() == ".json":
        return json.loads(text)
    return {}

def has_request_body(operation, parameters):
    if "requestBody" in operation:
        return True
    return any(str(p.get("in","")).lower() == "body" for p in parameters)

def merge_params(p1, p2):
    seen, merged = set(), []
    for group in [p1, p2]:
        if not isinstance(group, list): continue
        for p in group:
            if not isinstance(p, dict): continue
            key = (p.get("name"), p.get("in"))
            if key not in seen:
                seen.add(key)
                merged.append(p)
    return merged

def count_params(params):
    counts = {"path":0,"query":0,"header":0,"cookie":0,"body":0}
    for p in params:
        loc = str(p.get("in","")).lower()
        if loc in counts:
            counts[loc] += 1
    return counts

def has_auth(spec, op):
    if "security" in op:
        return int(bool(op["security"]))
    return int(bool(spec.get("security")))

def extract_response_flags(op):
    resp = op.get("responses", {})
    if not isinstance(resp, dict):
        return 0,0
    codes = [str(k) for k in resp.keys()]
    return int("401" in codes), int("403" in codes)

def ignore_path(path):
    p = path.lower()
    return any(x in p for x in IGNORE_PATHS)

# ---------- main ----------

def parse_file(fp):
    spec = load_spec(fp)
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return []

    rows = []

    for path, item in paths.items():
        if not isinstance(item, dict): continue
        if ignore_path(path): continue

        path_params = item.get("parameters", [])

        for method, op in item.items():
            if method.lower() not in HTTP_METHODS: continue
            if not isinstance(op, dict): continue

            op_params = op.get("parameters", [])
            merged = merge_params(path_params, op_params)
            counts = count_params(merged)

            has401, has403 = extract_response_flags(op)

            rows.append({
                "source_file": fp.as_posix(),
                "path": path.strip(),
                "method": method.upper(),
                "requires_auth": has_auth(spec, op),
                "has_request_body": int(has_request_body(op, merged)),
                "path_params": counts["path"],
                "query_params": counts["query"],
                "body_params": counts["body"],
                "total_parameters": len(merged),
                "resp_401": has401,
                "resp_403": has403
            })

    return rows

def collect_files():
    files = []
    for d in GOLD_DIRS + [API_GURU_DIR]:
        if not d.exists(): continue
        for f in d.rglob("*"):
            if f.suffix.lower() in VALID_SUFFIXES:
                files.append(f)
    return sorted(files)

def main():
    files = collect_files()
    print(f"Processing {len(files)} files")

    all_rows = []
    for f in files:
        try:
            all_rows.extend(parse_file(f))
        except:
            continue

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.drop_duplicates(["source_file","path","method"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Parsed {len(df)} endpoints")

if __name__ == "__main__":
    main()