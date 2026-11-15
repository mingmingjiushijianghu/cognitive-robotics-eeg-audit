"""Step 0 (V8): freeze the analysis plan before any AFFEC ERP result exists.

Copies 实验设计方案.md, config.py, common.py and every numbered step script into
preregistration/ and writes MANIFEST.json with SHA-256 hashes and a timestamp. Runs once;
the runner never refreezes. The author can upload this folder to OSF.
"""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE / "preregistration"


def main():
    if (HERE / "results" / "erp_trials.csv").exists():
        print("WARNING: ERP features already exist - this freeze is not blind to results")
    DST.mkdir(exist_ok=True)
    files = [HERE.parent / "实验设计方案.md", HERE / "config.py", HERE / "common.py"] + sorted(HERE.glob("[0-9][0-9]_*.py"))
    manifest = dict(frozen_at=dt.datetime.now().isoformat(timespec="seconds"), files={})
    for f in files:
        shutil.copy2(f, DST / f.name)
        manifest["files"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (DST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
