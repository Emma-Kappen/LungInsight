import json
import re
from pathlib import Path

root = Path.cwd()
clinical = root / "backend" / "Clinical"
out_dir = clinical / "artifacts" / "cancer_type_sex_smoking_baseline" / "integration_audit" / "stage_feature_inspection"
out_dir.mkdir(parents=True, exist_ok=True)

patterns = [
    re.compile(r"stage_at_baseline", re.I),
    re.compile(r"\bstage\b", re.I),
    re.compile(r"cancer-type-luad-lusc-v2", re.I),
    re.compile(r"sex-smoking-baseline", re.I),
]

skip_dirs = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "site-packages", "l-env"
}
extensions = {
    ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".ts", ".tsx", ".js", ".jsx", ".html", ".sql"
}

results = []
for path in root.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue
    try:
        rel = path.relative_to(root)
    except ValueError:
        continue
    if any(part in skip_dirs for part in rel.parts):
        continue
    if any(part.lower() in {"artifacts", "reports", "datasets", "data"} for part in rel.parts[:-1]):
        # Keep only the known Clinical integration locations and tests; avoid generated artifact trees.
        if "Clinical" not in rel.parts:
            continue
        if "tests" not in [p.lower() for p in rel.parts] and "schemas" not in [p.lower() for p in rel.parts] and "api" not in [p.lower() for p in rel.parts] and "tools" not in [p.lower() for p in rel.parts] and rel.name not in {"inference.py", "contract.py"}:
            continue
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        continue
    matches = []
    for number, line in enumerate(lines, start=1):
        tags = [p.pattern for p in patterns if p.search(line)]
        if tags:
            matches.append({
                "line": number,
                "patterns": tags,
                "text": line.strip()[:350]
            })
    if matches:
        results.append({
            "file": str(rel),
            "match_count": len(matches),
            "matches": matches
        })

results.sort(key=lambda x: (
    0 if any(s in x["file"].lower() for s in ("inference", "schema", "contract", "test", "api")) else 1,
    x["file"].lower()
))

report_path = out_dir / "stage_feature_references.json"
report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

for item in results:
    print(f"\n--- {item['file']} ({item['match_count']} matches) ---")
    for match in item["matches"]:
        print(f"{match['line']}: {match['text']}")

print(f"\nSaved full dependency inventory: {report_path}")
print(f"Files with matches: {len(results)}")
