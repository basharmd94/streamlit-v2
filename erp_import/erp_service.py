#!/usr/bin/env python
"""Runs the UtilityMart ERP importer jobs on a schedule, so ONE Windows service (NSSM) replaces four Task Scheduler entries.

Nothing listens here and no port is opened: the importer only calls OUT to the shop (https://medusa.utilitymart.shop/hooks/erp)
and to the Fixit database. This service is only the clock.

  every run_every_minutes   python erp_import.py --commit            shop feed -> ERP, status and stock back to the shop
  daily at fixit_time       python fixit_pull.py --commit            Fixit stock / price / cost -> um_fixit_item
  daily at price_time       python erp_import.py --job price --commit      100007 selling price from the source xsrate (a REPORT only
                                                                           until [pricing] apply = true)
  daily at products_time    python erp_import.py --job products --commit   barcode, price, cost, group, stock -> shop
  daily at po_time          python erp_import.py --job po --commit   open purchase orders for the day's orders

Each job is its own child process, one at a time (a crash or a hang in one job cannot take the service down; a job running
past job_timeout_minutes is killed). The importer's own database lock still stops any double run.
`--dry-run` leaves out --commit (everything is read and rolled back): use it for the first day on the server.
`--once` runs every job once now, in the order fixit, price, products, po, run, and exits (a smoke test; use with --dry-run).
Python 3.6+. Settings are the [schedule] section of erp_import.ini.
"""
import argparse
import configparser
import datetime as dt
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_LIVE = os.path.join(HERE, "logs", "service_state.json")
STATE_DRY = os.path.join(HERE, "logs", "service_state_dry.json")      # a dry run must never mark a daily job as done
POLL_SECONDS = 15
RETRY_DAILY_AFTER_MINUTES = 30        # a daily job that failed is tried again this much later (same day)
LOG = logging.getLogger("erp_service")


def read_schedule(path):
    cp = configparser.ConfigParser(interpolation=None)
    if not cp.read(path, encoding="utf-8"):
        raise SystemExit("config not found: %s" % path)

    def get(key, default):
        return cp.get("schedule", key, fallback=default).split("#")[0].strip() or default

    def hhmm(key, default):
        v = get(key, default)
        try:
            h, m = v.split(":")
            return dt.time(int(h), int(m))
        except ValueError:
            raise SystemExit("[schedule] %s must be HH:MM (24 hour), got %r" % (key, v))

    return {"run_every_minutes": max(1, int(get("run_every_minutes", "5"))),
            "fixit_time": hhmm("fixit_time", "05:30"),
            "price_time": hhmm("price_time", "05:45"),
            "products_time": hhmm("products_time", "06:00"),
            "po_time": hhmm("po_time", "23:30"),
            "job_timeout_minutes": max(1, int(get("job_timeout_minutes", "20")))}


def build_jobs(sched, commit):
    c = ["--commit"] if commit else []
    return [
        {"name": "fixit", "args": ["fixit_pull.py"] + c, "daily_at": sched["fixit_time"]},
        {"name": "price", "args": ["erp_import.py", "--job", "price"] + c, "daily_at": sched["price_time"]},
        {"name": "products", "args": ["erp_import.py", "--job", "products"] + c, "daily_at": sched["products_time"]},
        {"name": "po", "args": ["erp_import.py", "--job", "po"] + c, "daily_at": sched["po_time"]},
        {"name": "run", "args": ["erp_import.py"] + c, "every": dt.timedelta(minutes=sched["run_every_minutes"])},
    ]


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
    except OSError as e:
        LOG.warning("could not save %s: %s", path, e)


def is_due(job, state, now):
    last_try = state.get(job["name"] + "_tried")
    last_try = dt.datetime.strptime(last_try, "%Y-%m-%d %H:%M:%S") if last_try else None
    if "every" in job:
        return last_try is None or now - last_try >= job["every"]
    # daily: once the clock passes the time, until it has succeeded today; a failure is retried after a pause
    if state.get(job["name"] + "_ok") == now.date().isoformat():
        return False
    if now.time() < job["daily_at"]:
        return False
    return last_try is None or last_try.date() != now.date() or now - last_try >= dt.timedelta(minutes=RETRY_DAILY_AFTER_MINUTES)


def run_job(job, timeout_minutes):
    """Start one job as a child process. Returns the exit code (None if it had to be killed)."""
    cmd = [sys.executable] + job["args"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")        # Bangla item names must not break a cp1252 console
    LOG.info("start  %-8s %s", job["name"], " ".join(job["args"]))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout_minutes * 60, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        LOG.error("KILLED %-8s after %d minutes (job_timeout_minutes)", job["name"], timeout_minutes)
        return None
    except OSError as e:
        LOG.error("could not start %s: %s", job["name"], e)
        return -1
    secs = time.time() - t0
    if p.returncode == 0:
        LOG.info("done   %-8s %.0fs", job["name"], secs)
    else:
        tail = "\n".join((p.stdout or "").strip().splitlines()[-15:])
        LOG.error("FAILED %-8s exit %s after %.0fs. Last lines:\n%s", job["name"], p.returncode, secs, tail)
    return p.returncode


def setup_logging():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.handlers.TimedRotatingFileHandler(os.path.join(HERE, "logs", "erp_service.log"), when="midnight",
                                                   backupCount=30, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        root.addHandler(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(HERE, "erp_import.ini"))
    ap.add_argument("--dry-run", action="store_true", help="run every job without --commit (reads, writes nothing)")
    ap.add_argument("--once", action="store_true", help="run each job once now and exit (smoke test)")
    ap.add_argument("--show-schedule", action="store_true", help="print the schedule and exit")
    args = ap.parse_args()

    sched = read_schedule(args.config)
    jobs = build_jobs(sched, commit=not args.dry_run)
    if args.show_schedule:
        for j in jobs:
            print("%-8s %s" % (j["name"], ("every %d min" % (j["every"].seconds // 60)) if "every" in j
                               else "daily at %s" % j["daily_at"].strftime("%H:%M")))
        return 0

    setup_logging()
    LOG.info("erp_service started (%s, python %s, folder %s)", "DRY RUN" if args.dry_run else "LIVE: --commit",
             sys.version.split()[0], HERE)
    if args.once:
        order = ["fixit", "price", "products", "po", "run"]
        rc = 0
        for j in sorted(jobs, key=lambda x: order.index(x["name"])):
            r = run_job(j, sched["job_timeout_minutes"])
            rc = rc or (0 if r == 0 else 1)
        return rc

    state_path = STATE_DRY if args.dry_run else STATE_LIVE
    state = load_state(state_path)
    try:
        while True:
            now = dt.datetime.now().replace(microsecond=0)
            for job in jobs:                       # daily jobs come first in the list, the 5 minute feed last
                if not is_due(job, state, now):
                    continue
                state[job["name"] + "_tried"] = now.strftime("%Y-%m-%d %H:%M:%S")
                if run_job(job, sched["job_timeout_minutes"]) == 0 and "daily_at" in job:
                    state[job["name"] + "_ok"] = now.date().isoformat()
                save_state(state_path, state)
                now = dt.datetime.now().replace(microsecond=0)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        LOG.info("erp_service stopping (asked to stop)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
