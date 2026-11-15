"""Unattended pipeline runner for the embodied-emotion study (runs without Claude).

  python run_pipeline.py            run every step whose inputs are ready, then exit
  python run_pipeline.py --watch    keep going: re-check every 10 min (new data, edited
                                    scripts, finished GPU training) until stopped
  python run_pipeline.py --force 06 rerun step 06 (and everything downstream)
  python run_pipeline.py --status   print the state table

A step is up to date when its outputs exist, its script (+ common.py for EEG steps) has the
same SHA-256 as at its last success, and no upstream step finished after it. Edited scripts
therefore rerun themselves and their downstream steps automatically. Failed steps are retried
up to MAX_TRIES times per script version; independent steps keep running.
Step 03 (GPU training) runs in the background in parallel with the CPU steps.
State: logs/pipeline_state.json; readable summary: logs/pipeline_status.md;
per-step logs: logs/<step>.log. Only one runner at a time (logs/pipeline.lock).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
RES = HERE / "results"
STATE = LOGS / "pipeline_state.json"
LOCK = LOGS / "pipeline.lock"
PY = sys.executable
MAX_TRIES = 2
WATCH_EVERY = 600


def epochs_done():
    import config as C
    p = RES / "x2c_training_log.csv"
    return p.exists() and len(pd.read_csv(p)) >= C.X2C_EPOCHS


def affec_done():
    p = RES / "affec_preprocessing_qc.csv"
    if not p.exists():
        return False
    q = pd.read_csv(p)
    ok = q.error.isna() if "error" in q else pd.Series(True, index=q.index)
    return len(q) >= 70 and ok.all()


STEPS = [
    dict(id="00", script="00_freeze_prereg.py", deps=[], out=["preregistration/MANIFEST.json"], once=True),
    dict(id="01", script="01_miko_preprocess.py", deps=[], out=["results/miko_preprocessing_qc.csv"], common=True),
    dict(id="02", script="02_affec_preprocess.py", deps=[], out=["results/affec_preprocessing_qc.csv"], common=True, check=affec_done),
    dict(id="03", script="03_x2c_train.py", deps=[], out=["models/x2c_resnet18.pt"], check=epochs_done, background=True),
    dict(id="04", script="04_stim_features.py", deps=[], out=["results/affec_stim_features.csv"]),
    dict(id="05", script="05_erp_features.py", deps=["00", "01", "02", "04"], out=["results/erp_trials.csv"], common=True),
    dict(id="06", script="06_erp_stats.py", deps=["05"], out=["results/erp_stats_summary.md"]),
    dict(id="07", script="07_x2c_kinematics.py", deps=["03", "04"], out=["results/stim_kinematics.csv"]),
    dict(id="08", script="08_connectivity.py", deps=["01", "02"], out=["results/conn_stats.csv"]),
    dict(id="09", script="09_brain_behavior.py", deps=["05", "07", "08"], out=["results/bb_summary.md"]),
    dict(id="10", script="10_eed.py", deps=["05", "07", "08", "09"], out=["results/eed_summary.md"]),
    dict(id="12", script="12_robustness.py", deps=["01", "02", "04", "05"], out=["results/robust_summary.csv"]),
    dict(id="14", script="14_event_timing.py", deps=["01", "02"], out=["results/event_timing_summary.csv"], common=True),
    dict(id="15", script="15_model_calibration.py", deps=["05", "06"], out=["results/model_calibration_summary.csv"]),
    dict(id="11", script="11_report.py", deps=["06", "08", "09", "10", "12", "14", "15"], out=["results/REPORT.md"]),
    dict(id="13", script="13_paper_figures.py", deps=["11", "14", "15", "17", "18", "19"], out=["../submission/figures/Fig1.pdf"]),
    dict(id="17", script="17_timing_sensitivity.py", deps=["14"], out=["results/timing_sensitivity_summary.csv"]),
    dict(id="18", script="18_event_checks.py", deps=["04", "05"], out=["results/event_type_within_affec.csv"]),
    dict(id="19", script="19_late_lpp.py", deps=["02", "04"], out=["results/late_lpp.csv"], common=True),
    dict(id="16", script="16_paper_numbers.py", deps=["11", "14", "15", "17", "18", "19"], out=["../submission/manuscript/numbers.tex"]),
]
BY_ID = {s["id"]: s for s in STEPS}


def sha(step):
    h = hashlib.sha256((HERE / step["script"]).read_bytes())
    if step.get("common"):
        h.update((HERE / "common.py").read_bytes())
    return h.hexdigest()[:16]


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def outputs_ok(step):
    if not all((HERE / o).exists() for o in step["out"]):
        return False
    return step["check"]() if "check" in step else True


def up_to_date(step, st):
    s = st.get(step["id"], {})
    if s.get("status") != "done" or not outputs_ok(step):
        return False
    if step.get("once"):
        return True
    if s.get("sha") != sha(step):
        return False
    return all(st.get(d, {}).get("finished", "") <= s.get("finished", "") for d in step["deps"])


def running_elsewhere(script):
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if (p.pid != os.getpid() and "python" in (p.info["name"] or "").lower()
                    and any(Path(c).name == script for c in cmd[1:3])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def write_status(st, bg):
    rows = []
    for s in STEPS:
        x = st.get(s["id"], {})
        state = "running" if s["id"] in bg or running_elsewhere(s["script"]) else (
            "up to date" if up_to_date(s, st) else x.get("status", "pending"))
        rows.append(f"| {s['id']} | {s['script']} | {state} | {x.get('finished', '')} | {x.get('tries', 0)} | {x.get('error', '')[:80]} |")
    (LOGS / "pipeline_status.md").write_text(
        f"# 流水线状态（{now()}）\n\n| 步骤 | 脚本 | 状态 | 最近完成 | 尝试次数 | 最近错误 |\n|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n", encoding="utf-8")


def launch(step):
    log = open(LOGS / f"{step['id']}_{Path(step['script']).stem}.log", "a", encoding="utf-8")
    log.write(f"\n===== {now()} start {step['script']} =====\n"); log.flush()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return subprocess.Popen([PY, step["script"]], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env), log


def finish(step, proc, st):
    x = st.setdefault(step["id"], {})
    cur = sha(step)
    if x.get("sha_tried") != cur:
        x["tries"] = 0
    x["sha_tried"] = cur
    if proc.returncode == 0 and outputs_ok(step):
        x.update(status="done", sha=cur, finished=now(), error="")
    else:
        x["tries"] = x.get("tries", 0) + 1
        tail = (LOGS / f"{step['id']}_{Path(step['script']).stem}.log").read_text(encoding="utf-8", errors="replace")[-400:]
        x.update(status="failed", error=f"exit {proc.returncode}: " + tail.replace("\n", " ⏎ "))
    save_state(st)


def runnable(step, st, bg):
    if step["id"] in bg or up_to_date(step, st) or running_elsewhere(step["script"]):
        return False
    x = st.get(step["id"], {})
    if x.get("status") == "failed" and x.get("sha_tried") == sha(step) and x.get("tries", 0) >= MAX_TRIES:
        return False
    return all(up_to_date(BY_ID[d], st) for d in step["deps"])


def one_pass(bg):
    st = load_state()
    progressed = True
    while progressed:
        progressed = False
        for sid, (step, proc, log) in list(bg.items()):           # reap background jobs
            if proc.poll() is not None:
                log.close(); finish(step, proc, st); del bg[sid]; progressed = True
        for step in STEPS:
            if not runnable(step, st, bg):
                continue
            proc, log = launch(step)
            if step.get("background"):
                bg[step["id"]] = (step, proc, log)
                continue
            print(now(), "running", step["script"], flush=True)
            write_status(st, bg)
            proc.wait(); log.close()
            finish(step, proc, st)
            print(now(), step["script"], st[step["id"]]["status"], flush=True)
            progressed = True
            break                                                  # re-evaluate from the top
        write_status(st, bg)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--force", nargs="*", default=[])
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    if a.status:
        write_status(load_state(), {}); print((LOGS / "pipeline_status.md").read_text(encoding="utf-8")); return
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text())
            if psutil.pid_exists(pid) and "run_pipeline" in " ".join(psutil.Process(pid).cmdline()):
                print("another runner is active, pid", pid); return
        except (ValueError, psutil.Error):
            pass
    LOCK.write_text(str(os.getpid()))
    try:
        st = load_state()
        for f in a.force:
            st.pop(f, None)
        save_state(st)
        bg = {}
        while True:
            one_pass(bg)
            if not a.watch and not bg:
                break
            time.sleep(WATCH_EVERY if a.watch else 30)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
