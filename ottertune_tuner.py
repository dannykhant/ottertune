#!/usr/bin/env python3
"""
OtterTune Standalone Tuning Driver for adco-experiments DB Layer.
Uses OtterTune's core GP-BO algorithm directly (server/analysis/gp_tf.py).
Read-only integration: OtterTune algorithm code is untouched.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
import queue

import numpy as np
import psycopg2
from pyDOE import lhs
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler

# Ensure OtterTune server directory is in sys.path
local_server_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
if os.path.exists(local_server_dir) and local_server_dir not in sys.path:
    sys.path.insert(0, local_server_dir)
ottertune_server_dir = os.environ.get("PYTHONPATH", "/ottertune/server")
if ottertune_server_dir not in sys.path:
    sys.path.insert(0, ottertune_server_dir)

# Import OtterTune core algorithms (verbatim)
from analysis.gp_tf import GPRGD
from analysis.gp import GPRNP
from analysis.preprocessing import Bin, DummyEncoder
from analysis.constraints import ParamConstraintHelper

LOG = logging.getLogger("ottertune_tuner")

# OtterTune default hyperparameters
DEFAULT_PARAMS = {
    'GPR_USE_GPFLOW': False,
    'GPR_LENGTH_SCALE': 2.0,
    'GPR_MAGNITUDE': 1.0,
    'GPR_RIDGE': 1.0,
    'GPR_MAX_TRAIN_SIZE': 7000,
    'GPR_BATCH_SIZE': 3000,
    'TF_NUM_THREADS': 4,
    'GPR_LEARNING_RATE': 0.01,
    'GPR_EPSILON': 1e-6,
    'GPR_MAX_ITER': 100,
    'GPR_SIGMA_MULTIPLIER': 3.0,
    'GPR_MU_MULTIPLIER': 1.0,
    'GPR_EPS': 0.001,
    'NUM_SAMPLES': 30,
    'TOP_NUM_CONFIG': 10,
    'INIT_FLIP_PROB': 0.3,
    'FLIP_PROB_DECAY': 0.5,
}

UNITS_MULTIPLIER = {
    'B': 1,
    'kB': 1024,
    'MB': 1024 ** 2,
    'GB': 1024 ** 3,
    'TB': 1024 ** 4,
    '8kB': 8 * 1024,
    '16kB': 16 * 1024,
    '32kB': 32 * 1024,
    '64kB': 64 * 1024,
    '128kB': 128 * 1024,
    '256kB': 256 * 1024,
    '512kB': 512 * 1024,
    'ms': 1,
    's': 1000,
    'min': 60000,
    'us': 1,
}


def get_db_connection(args, dbname=None):
    """Create a database connection with auto-retry."""
    db = dbname or args.db_name
    for _ in range(30):
        try:
            conn = psycopg2.connect(
                host=args.db_host,
                port=args.db_port,
                dbname=db,
                user=args.db_user,
                password=args.db_password,
                connect_timeout=5
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            time.sleep(2)
    raise ConnectionError(f"Could not connect to database {db} on {args.db_host}:{args.db_port}")


def collect_metrics(conn):
    """Collect PostgreSQL runtime performance metrics."""
    metrics = {}
    with conn.cursor() as cur:
        cur.execute("SHOW server_version_num")
        server_version = int(cur.fetchone()[0])

        # pg_stat_database metrics
        cur.execute("""
            SELECT
                COALESCE(sum(xact_commit), 0),
                COALESCE(sum(xact_rollback), 0),
                COALESCE(sum(blks_read), 0),
                COALESCE(sum(blks_hit), 0),
                COALESCE(sum(tup_returned), 0),
                COALESCE(sum(tup_fetched), 0),
                COALESCE(sum(tup_inserted), 0),
                COALESCE(sum(tup_updated), 0),
                COALESCE(sum(tup_deleted), 0),
                COALESCE(sum(temp_files), 0),
                COALESCE(sum(temp_bytes), 0),
                COALESCE(sum(deadlocks), 0)
            FROM pg_stat_database
        """)
        row = cur.fetchone()
        db_cols = [
            'xact_commit', 'xact_rollback', 'blks_read', 'blks_hit',
            'tup_returned', 'tup_fetched', 'tup_inserted', 'tup_updated',
            'tup_deleted', 'temp_files', 'temp_bytes', 'deadlocks'
        ]
        for col, val in zip(db_cols, row):
            metrics[col] = float(val or 0)

        # Checkpointer / bgwriter metrics
        if server_version >= 170000:
            cur.execute("""
                SELECT
                    COALESCE(num_timed, 0),
                    COALESCE(num_requested, 0),
                    COALESCE(write_time, 0),
                    COALESCE(sync_time, 0),
                    COALESCE(buffers_written, 0)
                FROM pg_stat_checkpointer
            """)
            cp_row = cur.fetchone() or (0, 0, 0, 0, 0)
            metrics['checkpoints_timed'] = float(cp_row[0])
            metrics['checkpoints_req'] = float(cp_row[1])
            metrics['checkpoint_write_time'] = float(cp_row[2])
            metrics['checkpoint_sync_time'] = float(cp_row[3])
            metrics['buffers_checkpoint'] = float(cp_row[4])

            cur.execute("""
                SELECT
                    COALESCE(buffers_clean, 0),
                    COALESCE(maxwritten_clean, 0),
                    COALESCE(buffers_alloc, 0)
                FROM pg_stat_bgwriter
            """)
            bg_row = cur.fetchone() or (0, 0, 0)
            metrics['buffers_clean'] = float(bg_row[0])
            metrics['maxwritten_clean'] = float(bg_row[1])
            metrics['buffers_alloc'] = float(bg_row[2])
        else:
            cur.execute("""
                SELECT
                    COALESCE(checkpoints_timed, 0),
                    COALESCE(checkpoints_req, 0),
                    COALESCE(checkpoint_write_time, 0),
                    COALESCE(checkpoint_sync_time, 0),
                    COALESCE(buffers_checkpoint, 0),
                    COALESCE(buffers_clean, 0),
                    COALESCE(maxwritten_clean, 0),
                    COALESCE(buffers_alloc, 0)
                FROM pg_stat_bgwriter
            """)
            bg_row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0, 0)
            metrics['checkpoints_timed'] = float(bg_row[0])
            metrics['checkpoints_req'] = float(bg_row[1])
            metrics['checkpoint_write_time'] = float(bg_row[2])
            metrics['checkpoint_sync_time'] = float(bg_row[3])
            metrics['buffers_checkpoint'] = float(bg_row[4])
            metrics['buffers_clean'] = float(bg_row[5])
            metrics['maxwritten_clean'] = float(bg_row[6])
            metrics['buffers_alloc'] = float(bg_row[7])

    return metrics


def collect_knobs(conn, knob_names):
    """Collect current knob values in normalized unit space."""
    knobs = {}
    with conn.cursor() as cur:
        cur.execute("SELECT name, setting, unit, vartype FROM pg_settings WHERE name = ANY(%s)", (knob_names,))
        rows = cur.fetchall()
        for name, setting, unit, vartype in rows:
            if vartype in ('integer', 'real'):
                try:
                    val = int(setting)
                except ValueError:
                    val = float(setting)
                if unit and unit in UNITS_MULTIPLIER:
                    val = val * UNITS_MULTIPLIER[unit]
                knobs[name] = val
            elif vartype == 'bool':
                knobs[name] = 1 if setting.lower() in ('on', 'true', 'yes', '1') else 0
            else:
                knobs[name] = setting
    return knobs


def get_knob_native_bounds(conn):
    """Get native units and limits directly from pg_settings."""
    bounds = {}
    with conn.cursor() as cur:
        cur.execute("SELECT name, unit, min_val, max_val, context FROM pg_settings")
        for name, unit, min_val, max_val, context in cur.fetchall():
            def parse_num(v):
                if v is None:
                    return None
                try:
                    f = float(v)
                    return int(f) if f.is_integer() else f
                except (ValueError, TypeError):
                    return None
            bounds[name] = {
                'unit': unit,
                'min': parse_num(min_val),
                'max': parse_num(max_val),
                'context': context
            }
    return bounds


def gen_lhs_samples(knob_catalog, nsamples):
    """Generate initial Latin Hypercube Samples (from OtterTune async_tasks)."""
    names = sorted(list(knob_catalog.keys()))
    minvals = []
    maxvals = []
    types = []

    for name in names:
        info = knob_catalog[name]
        min_v = float(info['minval']) if info['minval'] is not None else 0.0
        max_v = float(info['maxval']) if info['maxval'] is not None else 1.0
        minvals.append(min_v)
        maxvals.append(max_v)
        types.append(info['vartype'])

    nfeats = len(names)
    samples = lhs(nfeats, samples=nsamples, criterion='maximin')
    scales = np.array(maxvals) - np.array(minvals)
    for fi in range(nfeats):
        samples[:, fi] = uniform(loc=minvals[fi], scale=scales[fi]).ppf(samples[:, fi])

    lhs_samples = []
    for si in range(nsamples):
        sample_dict = {}
        for fi in range(nfeats):
            val = samples[si][fi]
            # vartype 2 is INTEGER, 3 is REAL, 4 is BOOL, 5 is ENUM
            if types[fi] in (2, 4):
                sample_dict[names[fi]] = int(round(val))
            elif types[fi] == 3:
                sample_dict[names[fi]] = float(val)
            else:
                sample_dict[names[fi]] = int(round(val))
        lhs_samples.append(sample_dict)

    random.shuffle(lhs_samples)
    return lhs_samples


def recommend_gpr(history, knob_catalog, params):
    """
    Run OtterTune's GPRGD Bayesian Optimization recommendation.
    Verbatim logic from OtterTune async_tasks.configuration_recommendation.
    """
    knob_names = sorted(list(knob_catalog.keys()))
    X_matrix = np.array([[h['knobs'][k] for k in knob_names] for h in history], dtype=np.float64)

    # Compute target metric: transaction throughput or commit count change
    y_matrix = []
    for h in history:
        # Check if throughput metric is reported by workload output
        wl_res = h.get('workload_result', {})
        tps = wl_res.get('throughput') or wl_res.get('tps')
        if tps is not None:
            y_val = float(tps)
        else:
            # Fallback to database commit delta
            delta_commits = h['metrics_after'].get('xact_commit', 0) - h['metrics_before'].get('xact_commit', 0)
            delta_hits = h['metrics_after'].get('blks_hit', 0) - h['metrics_before'].get('blks_hit', 0)
            y_val = float(delta_commits if delta_commits > 0 else delta_hits)
        y_matrix.append([y_val])

    y_matrix = np.array(y_matrix, dtype=np.float64)

    # Scaled knob bounds
    X_min = np.array([float(knob_catalog[k]['minval']) if knob_catalog[k]['minval'] is not None else 0.0 for k in knob_names], dtype=np.float64)
    X_max = np.array([float(knob_catalog[k]['maxval']) if knob_catalog[k]['maxval'] is not None else 1.0 for k in knob_names], dtype=np.float64)

    # Standardize X using history augmented with boundary points to ensure non-zero variance
    X_scaler = StandardScaler()
    X_scaler.fit(np.vstack([X_matrix, X_min.reshape(1, -1), X_max.reshape(1, -1)]))
    X_scaled = X_scaler.transform(X_matrix)

    # Standardize y
    y_std = np.std(y_matrix)
    if y_std > 1e-6:
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_matrix)
    else:
        y_scaled = np.zeros_like(y_matrix)

    # OtterTune minimizes the acquisition function, so maximize throughput -> minimize -y
    y_scaled = -y_scaled

    X_min_s = X_scaler.transform(X_min.reshape(1, -1))[0]
    X_max_s = X_scaler.transform(X_max.reshape(1, -1))[0]

    # Generate candidate samples for BO exploration (random + best observed points)
    num_samples = params['NUM_SAMPLES']
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    for i in range(X_scaled.shape[1]):
        X_samples[:, i] = np.random.rand(num_samples) * (X_max_s[i] - X_min_s[i]) + X_min_s[i]

    # Add top observed configs with small epsilon perturbation
    q = queue.PriorityQueue()
    for x in range(y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))

    i = 0
    while i < params['TOP_NUM_CONFIG']:
        try:
            item = q.get_nowait()
            eps = abs(params['GPR_EPS'])
            dist = sum(np.square(X_max_s - X_scaled[item[1]]))
            if dist < 0.001:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] - eps))
            else:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] + eps))
            i += 1
        except queue.Empty:
            break

    # Execute OtterTune GPRGD model
    model = GPRGD(
        length_scale=params['GPR_LENGTH_SCALE'],
        magnitude=params['GPR_MAGNITUDE'],
        max_train_size=params['GPR_MAX_TRAIN_SIZE'],
        batch_size=params['GPR_BATCH_SIZE'],
        num_threads=params['TF_NUM_THREADS'],
        learning_rate=params['GPR_LEARNING_RATE'],
        epsilon=params['GPR_EPSILON'],
        max_iter=params['GPR_MAX_ITER'],
        sigma_multiplier=params['GPR_SIGMA_MULTIPLIER'],
        mu_multiplier=params['GPR_MU_MULTIPLIER'],
        ridge=params['GPR_RIDGE']
    )
    model.fit(X_scaled, y_scaled, X_min_s, X_max_s)
    res = model.predict(X_samples)

    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    best_config = X_scaler.inverse_transform(best_config)

    # Clamp within original bounds
    best_config = np.minimum(best_config, X_max)
    best_config = np.maximum(best_config, X_min)

    recommendation = {}
    for i, name in enumerate(knob_names):
        info = knob_catalog[name]
        if info['vartype'] in (2, 4):  # INTEGER, BOOL
            recommendation[name] = int(round(best_config[i]))
        elif info['vartype'] == 3:  # REAL
            recommendation[name] = round(float(best_config[i]), 4)
        else:
            recommendation[name] = int(round(best_config[i]))

    return recommendation


def apply_knobs(conn, recommendation, native_bounds, knob_catalog, restart_cmd):
    """Apply knobs to PostgreSQL via ALTER SYSTEM SET and reload/restart."""
    needs_restart = False

    with conn.cursor() as cur:
        for name, value in recommendation.items():
            info = native_bounds.get(name)
            if not info:
                continue

            catalog_info = knob_catalog.get(name, {})
            vartype = catalog_info.get('vartype')
            enumvals_str = catalog_info.get('enumvals')

            if vartype == 5 and enumvals_str:
                enumvals = enumvals_str.split(',')
                idx = int(round(value)) if isinstance(value, (int, float)) else 0
                idx = max(0, min(idx, len(enumvals) - 1))
                native_value = enumvals[idx]
            elif isinstance(value, (int, float)):
                unit = info.get('unit')
                factor = UNITS_MULTIPLIER.get(unit, 1)
                native_value = value / factor
                if info['min'] is not None and info['max'] is not None:
                    native_value = min(max(native_value, info['min']), info['max'])
                if isinstance(native_value, float) and native_value.is_integer():
                    native_value = int(native_value)
            else:
                native_value = value

            if info.get('context') == 'postmaster':
                needs_restart = True

            try:
                cur.execute(f"ALTER SYSTEM SET {name} = %s", (native_value,))
            except Exception as e:
                LOG.warning("Could not set %s to %s: %s", name, native_value, e)

    if needs_restart and restart_cmd:
        LOG.info("Postmaster context knob changed, restarting database...")
        conn.close()
        subprocess.run(restart_cmd, shell=True, check=True)
        time.sleep(5)
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_reload_conf()")
        LOG.info("Configuration reloaded via pg_reload_conf()")


def run_tuning_loop(args):
    """Main OtterTune Closed-Loop Optimization."""
    if args.knob_catalog and os.path.exists(args.knob_catalog):
        with open(args.knob_catalog) as f:
            knob_catalog = json.load(f)
    else:
        from generate_knob_catalog import generate_catalog
        LOG.info("Generating hardware-scaled knob catalog for %s (RAM: %.1fGB, CPUs: %d)",
                 args.dbms, args.memory_gb, args.cpu_cores)
        knob_catalog = generate_catalog(
            dbms=args.dbms,
            memory_gb=args.memory_gb,
            cpu_cores=args.cpu_cores,
            storage_gb=args.storage_gb
        )

    knob_names = sorted(list(knob_catalog.keys()))
    params = DEFAULT_PARAMS
    history = []
    lhs_pool = gen_lhs_samples(knob_catalog, max(args.warmup, 10))

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark_dir = os.path.join(args.output_dir, "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)

    LOG.info("OtterTune starting for %s (iterations=%d, warmup=%d)", args.workload_name, args.iterations, args.warmup)

    best_metric = -float('inf')
    best_config = None

    for i in range(args.iterations):
        iter_dir = os.path.join(args.output_dir, f"iter_{i:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        LOG.info("-------------------<< Iteration %d / %d >>-------------------", i + 1, args.iterations)

        conn = get_db_connection(args)
        native_bounds = get_knob_native_bounds(conn)
        current_knobs = collect_knobs(conn, knob_names)

        # 1. Choose next configuration (LHS warmup or GPR recommendation)
        if i < args.warmup and lhs_pool:
            rec = lhs_pool.pop()
            LOG.info("Using LHS sample for exploration")
        else:
            LOG.info("Computing recommendation via OtterTune GPRGD...")
            rec = recommend_gpr(history, knob_catalog, params)

        # 2. Apply configuration
        apply_knobs(conn, rec, native_bounds, knob_catalog, args.restart_cmd)
        conn = get_db_connection(args)

        # 3. Collect metrics before workload
        metrics_before = collect_metrics(conn)
        conn.close()

        # 4. Run workload iteration if script provided
        workload_res = {}
        if args.workload_cmd:
            LOG.info("Executing workload: %s", args.workload_cmd)
            t0 = time.time()
            proc = subprocess.run(args.workload_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - t0
            workload_res = {
                'elapsed_sec': elapsed,
                'returncode': proc.returncode,
                'output': proc.stdout
            }
            # Attempt to extract throughput (TPS) from stdout
            for line in proc.stdout.splitlines():
                line_clean = line.strip()
                # Check CSV-like TOTAL row: e.g. TOTAL,10000,4249.59,2353.16
                if line_clean.startswith("TOTAL,") or line_clean.startswith("total,"):
                    parts = line_clean.split(",")
                    if len(parts) >= 4:
                        try:
                            workload_res['throughput'] = float(parts[3])
                            break
                        except ValueError:
                            pass
                # Check whitespace-formatted TOTAL row: e.g. TOTAL 5 24.39 204.98 txn/s
                elif line_clean.startswith("TOTAL") and ("txn/s" in line_clean or len(line_clean.split()) >= 4):
                    tokens = line_clean.replace("txn/s", "").split()
                    if len(tokens) >= 4:
                        try:
                            workload_res['throughput'] = float(tokens[3])
                            break
                        except ValueError:
                            pass
                elif any(k in line for k in ["Throughput:", "Throughput (requests/sec):", "TPS:", "transactions/s:"]):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            workload_res['throughput'] = float(parts[1].strip().split()[0])
                            break
                        except ValueError:
                            pass
            if 'throughput' in workload_res:
                LOG.info("Measured workload throughput: %.2f TPS", workload_res['throughput'])
            else:
                LOG.warning("Could not parse throughput from workload output")

        # 5. Collect metrics after workload
        conn = get_db_connection(args)
        metrics_after = collect_metrics(conn)
        conn.close()

        # 6. Record history
        record = {
            'iteration': i,
            'knobs': rec,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'workload_result': workload_res
        }
        history.append(record)

        # Track best configuration
        tps = workload_res.get('throughput')
        if tps is None:
            tps = metrics_after.get('xact_commit', 0) - metrics_before.get('xact_commit', 0)

        if tps > best_metric:
            best_metric = tps
            best_config = rec
            LOG.info("New best performance achieved: %s", best_metric)

        # Save artifacts
        with open(os.path.join(iter_dir, "knobs.json"), "w") as f:
            json.dump(rec, f, indent=2)
        with open(os.path.join(iter_dir, "metrics_before.json"), "w") as f:
            json.dump(metrics_before, f, indent=2)
        with open(os.path.join(iter_dir, "metrics_after.json"), "w") as f:
            json.dump(metrics_after, f, indent=2)
        with open(os.path.join(iter_dir, "workload_result.json"), "w") as f:
            json.dump(workload_res, f, indent=2)

    # Save best configuration and summary
    best_file = os.path.join(args.output_dir, "best_configuration.json")
    with open(best_file, "w") as f:
        json.dump(best_config, f, indent=2)
    LOG.info("Tuning complete. Best configuration saved to %s", best_file)

    # Apply best configuration to the database for final evaluation
    if best_config:
        conn = get_db_connection(args)
        native_bounds = get_knob_native_bounds(conn)
        apply_knobs(conn, best_config, native_bounds, knob_catalog, args.restart_cmd)
        conn.close()
        LOG.info("Applied best configuration to database.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="OtterTune Database Tuner")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="postgres")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="postgres")
    parser.add_argument("--workload-name", default="custom")
    parser.add_argument("--workload-cmd", default=None, help="Command to run during each tuning iteration")
    parser.add_argument("--restart-cmd", default=None, help="Command to restart DB container if needed")
    parser.add_argument("--dbms", default="postgres", choices=["postgres", "postgresql", "mysql"])
    parser.add_argument("--memory-gb", type=float, default=2.0, help="Target system RAM in GB (default: 2.0)")
    parser.add_argument("--cpu-cores", type=int, default=2, help="Target system CPU cores (default: 2)")
    parser.add_argument("--storage-gb", type=float, default=10.0, help="Target system storage in GB (default: 10.0)")
    parser.add_argument("--knob-catalog", default="knob_catalog.json")
    parser.add_argument("--output-dir", default="results/db_layer/ottertune")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    run_tuning_loop(args)
