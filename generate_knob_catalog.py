#!/usr/bin/env python3
"""
OtterTune Knob Catalog Generator.
Extracts ranked tunable DBMS knobs from OtterTune's fixtures and scales min/max
value boundaries according to target hardware specifications using OtterTune's
set_default_knobs() algorithm (server/website/website/set_default_knobs.py).
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("generate_knob_catalog")

GB = 1024 ** 3
DEFAULT_SESSION_NUM = 50.0

# Hardware percentage scaling factors from OtterTune set_default_knobs.py
CPU_PERCENT = 2.0
MEMORY_PERCENT = 0.8
STORAGE_PERCENT = 0.8
DEFAULT_MAXVAL = 192 * GB


def get_fixtures_dir(repo_dir=None):
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(repo_dir, "server", "website", "website", "fixtures")
    if not os.path.exists(fixtures_dir):
        raise FileNotFoundError(f"Fixtures directory not found at {fixtures_dir}")
    return fixtures_dir


def load_knob_fixtures(dbms, fixtures_dir):
    """Load the appropriate knob catalog fixture for the DBMS."""
    if dbms in ("postgres", "postgresql"):
        fixture_file = os.path.join(fixtures_dir, "postgres-96_knobs.json")
    elif dbms == "mysql":
        fixture_file = os.path.join(fixtures_dir, "mysql-57_knobs.json")
    else:
        raise ValueError(f"Unsupported DBMS: {dbms}. Supported: postgres, mysql")

    with open(fixture_file, "r") as f:
        return json.load(f)


def generate_catalog(dbms="postgres", memory_gb=2.0, cpu_cores=2, storage_gb=10.0, repo_dir=None):
    """
    Generate knob catalog dictionary using OtterTune's set_default_knobs logic.
    """
    fixtures_dir = get_fixtures_dir(repo_dir)
    data = load_knob_fixtures(dbms, fixtures_dir)

    total_memory = memory_gb * GB
    total_storage = storage_gb * GB

    tunable_knobs = {}
    for entry in data:
        fields = entry["fields"]
        if fields.get("tunable") is not True:
            continue

        name = fields["name"]
        if name.startswith("global."):
            clean_name = name[len("global."):]
        else:
            clean_name = name

        vartype = int(fields["vartype"])
        resource = int(fields.get("resource", 4))
        raw_min = fields.get("minval")
        raw_max = fields.get("maxval")

        # VarType: 1=STRING, 2=INTEGER, 3=REAL, 4=BOOL, 5=ENUM
        if vartype == 5:  # ENUM
            enumvals = fields.get("enumvals", "").split(",")
            minval = 0
            maxval = len(enumvals) - 1
            if clean_name == "wal_sync_method":
                # Limit to POSIX-safe sync methods: 0=fsync, 1=fdatasync
                maxval = min(1, maxval)
        elif vartype == 4:  # BOOL
            minval = 0
            maxval = 1
        elif vartype in (2, 3):  # INTEGER, REAL
            vtype = int if vartype == 2 else float

            # --- Dynamic Proportional Hardware Rules ---
            if clean_name == "shared_buffers":
                minval = max(128 * (1024 ** 2), int(total_memory * 0.10))
                maxval = max(minval, int(total_memory * 0.40))
            elif clean_name == "effective_cache_size":
                minval = int(total_memory * 0.25)
                maxval = max(minval, int(total_memory * 0.75))
            elif clean_name == "wal_buffers":
                minval = 4 * (1024 ** 2)
                maxval = min(64 * (1024 ** 2), max(16 * (1024 ** 2), int(total_memory * 0.03)))
            elif clean_name == "work_mem":
                minval = 4 * (1024 ** 2)
                maxval = max(minval, int((total_memory * 0.25) / DEFAULT_SESSION_NUM))
            elif clean_name == "temp_buffers":
                minval = 4 * (1024 ** 2)
                maxval = max(minval, int((total_memory * 0.20) / DEFAULT_SESSION_NUM))
            elif clean_name == "maintenance_work_mem":
                minval = min(64 * (1024 ** 2), int(total_memory * 0.05))
                maxval = max(minval, min(2 * GB, int(total_memory * 0.25)))
            elif clean_name == "max_wal_size":
                minval = 1 * GB
                maxval = max(minval, min(64 * GB, int(total_storage * 0.50)))
            elif clean_name == "min_wal_size":
                minval = 80 * (1024 ** 2)
                maxval = max(minval, min(4 * GB, int(total_storage * 0.20)))
            elif clean_name == "max_worker_processes":
                minval = 1
                maxval = max(1, cpu_cores)
            elif clean_name == "max_parallel_workers_per_gather":
                minval = 0
                maxval = max(1, cpu_cores // 2)
            elif clean_name == "effective_io_concurrency":
                minval = 1
                maxval = min(1024, cpu_cores * 100)
            elif clean_name == "seq_page_cost":
                minval = 1.0
                maxval = 1.2
            elif clean_name == "random_page_cost":
                minval = 1.1
                maxval = 4.0
            elif clean_name in ("join_collapse_limit", "from_collapse_limit"):
                minval = 8
                maxval = 20
            elif clean_name == "default_statistics_target":
                minval = 50
                maxval = 500
            elif clean_name in ("bgwriter_delay", "wal_writer_delay"):
                minval = 10
                maxval = 500
            elif clean_name == "deadlock_timeout":
                minval = 100
                maxval = 2000
            elif clean_name == "commit_delay":
                minval = 0
                maxval = 1000
            else:
                minval = vtype(raw_min) if raw_min is not None else 0
                maxval = vtype(raw_max) if raw_max is not None else DEFAULT_MAXVAL

            if maxval < minval:
                maxval = minval * 2

            minval = vtype(minval)
            maxval = vtype(maxval)
        else:
            minval = 0
            maxval = 1

        tunable_knobs[clean_name] = {
            "vartype": vartype,
            "unit": int(fields.get("unit", 3)),
            "minval": minval,
            "maxval": maxval,
            "default": fields.get("default"),
            "enumvals": fields.get("enumvals"),
            "context": fields.get("context", "user"),
            "scope": fields.get("scope", "global"),
        }

    return tunable_knobs


def main():
    parser = argparse.ArgumentParser(description="Generate hardware-scaled OtterTune knob catalog")
    parser.add_argument("--dbms", default="postgres", choices=["postgres", "postgresql", "mysql"])
    parser.add_argument("--memory-gb", type=float, default=2.0, help="Target system RAM in GB (default: 2.0)")
    parser.add_argument("--cpu-cores", type=int, default=2, help="Target system CPU cores (default: 2)")
    parser.add_argument("--storage-gb", type=float, default=10.0, help="Target system storage in GB (default: 10.0)")
    parser.add_argument("--output", default="knob_catalog.json", help="Output JSON path (default: knob_catalog.json)")
    args = parser.parse_args()

    LOG.info("Generating OtterTune knob catalog for %s (RAM: %.1fGB, CPUs: %d)", args.dbms, args.memory_gb, args.cpu_cores)
    catalog = generate_catalog(
        dbms=args.dbms,
        memory_gb=args.memory_gb,
        cpu_cores=args.cpu_cores,
        storage_gb=args.storage_gb
    )

    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2)

    LOG.info("Generated %d tunable knobs saved to %s", len(catalog), args.output)


if __name__ == "__main__":
    main()
