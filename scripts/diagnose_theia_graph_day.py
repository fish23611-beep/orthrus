#!/usr/bin/env python3
"""
diagnose_theia_graph_day.py

Read-only diagnostic tool to investigate why graph_2 is missing from THEIA_E3 build.

This script is a temporary investigation tool. It does NOT:
- Write to the database
- Write graph artifacts
- Modify configuration
- Trigger preprocessing
- Delete existing artifacts
- Run model training
- Commit or push

Usage:
    python scripts/diagnose_theia_graph_day.py THEIA_E3 --day 2
    python scripts/diagnose_theia_graph_day.py THEIA_E3 --day 2 --config config/orthrus.yml
    python scripts/diagnose_theia_graph_day.py THEIA_E3 --day 2 --verbose
"""

import argparse
import sys
import os
from datetime import datetime

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from config import (
    get_yml_cfg,
    get_runtime_required_args,
    DATASET_DEFAULT_CONFIG,
    rel2id,
)
from provnet_utils import (
    init_database_connection,
    datetime_to_ns_time_US,
    ns_time_to_datetime_US,
    log as provnet_log,
)
from graph_construction.build_orthrus_graphs import (
    stream_event_table,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for THEIA graph day analysis (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (e.g., THEIA_E3)",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=2,
        help="Day number to diagnose (default: 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    return parser


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def safe_log(msg, verbose_only=False):
    """Log message, optionally suppressing non-verbose output."""
    if verbose_only:
        return
    print(msg)


def format_ns_timestamp(ns):
    """Format nanosecond timestamp to human-readable string."""
    try:
        return ns_time_to_datetime_US(ns)
    except Exception:
        return f"{ns} (unparseable)"


def get_db_identity(cur):
    """Get database identity information."""
    results = {}

    # Database name
    cur.execute("SELECT current_database();")
    results["current_database"] = cur.fetchone()[0]

    # PostgreSQL version
    cur.execute("SELECT version();")
    results["version"] = cur.fetchone()[0]

    # Server version (numeric)
    cur.execute("SHOW server_version_num;")
    results["server_version_num"] = cur.fetchone()[0]

    # Port
    try:
        cur.execute("SHOW port;")
        results["port"] = cur.fetchone()[0]
    except Exception:
        results["port"] = "unknown"

    # Session timezone
    try:
        cur.execute("SHOW TimeZone;")
        results["session_timezone"] = cur.fetchone()[0]
    except Exception:
        results["session_timezone"] = "unknown"

    # Data directory (may fail due to permissions)
    try:
        cur.execute("SHOW data_directory;")
        results["data_directory"] = cur.fetchone()[0]
    except Exception:
        results["data_directory"] = "<unavailable>"

    return results


def check_event_table_exists(cur):
    """Check if event_table exists and return row count."""
    try:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'event_table';
        """)
        exists = cur.fetchone()[0] > 0
        return exists, None
    except Exception as e:
        return False, str(e)


def get_event_table_global_stats(cur):
    """Get global statistics for event_table."""
    stats = {}
    try:
        cur.execute("SELECT COUNT(*) FROM event_table;")
        stats["total_rows"] = cur.fetchone()[0]
    except Exception as e:
        stats["total_rows"] = None
        stats["error"] = str(e)
        return stats

    try:
        cur.execute("SELECT MIN(timestamp_rec), MAX(timestamp_rec) FROM event_table;")
        min_ts, max_ts = cur.fetchone()
        stats["min_timestamp_rec"] = min_ts
        stats["max_timestamp_rec"] = max_ts
        stats["min_timestamp_readable"] = format_ns_timestamp(min_ts) if min_ts else None
        stats["max_timestamp_readable"] = format_ns_timestamp(max_ts) if max_ts else None
    except Exception as e:
        stats["min_timestamp_rec"] = None
        stats["max_timestamp_rec"] = None
        stats["error"] = str(e)

    return stats


def count_day_raw_events(cur, year_month, day):
    """Count raw events for a specific day using direct SQL.

    Uses the same timestamp boundary semantics as production code:
    timestamp_rec > start AND timestamp_rec < end
    """
    date_start = f"{year_month}-{day} 00:00:00"
    date_stop = f"{year_month}-{day + 1} 00:00:00"

    start_ns = datetime_to_ns_time_US(date_start)
    end_ns = datetime_to_ns_time_US(date_stop)

    sql = f"""
        SELECT COUNT(*) FROM event_table
        WHERE timestamp_rec > {start_ns} AND timestamp_rec < {end_ns};
    """
    cur.execute(sql)
    return cur.fetchone()[0], start_ns, end_ns


def get_day_operation_distribution(cur, year_month, day):
    """Get operation distribution for a specific day."""
    date_start = f"{year_month}-{day} 00:00:00"
    date_stop = f"{year_month}-{day + 1} 00:00:00"

    start_ns = datetime_to_ns_time_US(date_start)
    end_ns = datetime_to_ns_time_US(date_stop)

    sql = f"""
        SELECT operation, COUNT(*) as cnt
        FROM event_table
        WHERE timestamp_rec > {start_ns} AND timestamp_rec < {end_ns}
        GROUP BY operation
        ORDER BY cnt DESC;
    """
    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def filter_supported_relations(events, supported_relations):
    """Filter events to only those with supported relations."""
    return [e for e in events if e[2] in supported_relations]


def stream_filtered_semantic_batches(cur, sql, include_edge_type, batch_size=1024):
    """
    Streaming version of filtered semantic batches.

    Replicates the production logic from build_orthrus_graphs.py.
    """
    batch = []
    for event in stream_event_table(cur, sql, batch_size):
        if event[2] not in include_edge_type:
            continue
        batch.append(event)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_production_streaming_raw(cur, start_ns, end_ns):
    """Count raw events using production stream_event_table."""
    sql = f"""
        SELECT * FROM event_table
        WHERE timestamp_rec > {start_ns} AND timestamp_rec < {end_ns}
        ORDER BY timestamp_rec, event_uuid;
    """
    count = 0
    for _ in stream_event_table(cur, sql, batch_size=8192):
        count += 1
    return count


def count_production_filtered_semantic(cur, start_ns, end_ns, include_edge_type):
    """Count semantic-filtered events using production streaming."""
    sql = f"""
        SELECT * FROM event_table
        WHERE timestamp_rec > {start_ns} AND timestamp_rec < {end_ns}
        ORDER BY timestamp_rec, event_uuid;
    """
    count = 0
    for _ in stream_filtered_semantic_batches(cur, sql, include_edge_type, batch_size=1024):
        count += len(_)
    return count


def get_sample_events(cur, year_month, day, limit=5):
    """Get sample events for a specific day."""
    date_start = f"{year_month}-{day} 00:00:00"
    date_stop = f"{year_month}-{day + 1} 00:00:00"

    start_ns = datetime_to_ns_time_US(date_start)
    end_ns = datetime_to_ns_time_US(date_stop)

    sql = f"""
        SELECT src_node, src_index_id, operation, dst_node, dst_index_id,
               event_uuid, timestamp_rec, id
        FROM event_table
        WHERE timestamp_rec > {start_ns} AND timestamp_rec < {end_ns}
        ORDER BY timestamp_rec
        LIMIT {limit};
    """
    cur.execute(sql)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Diagnostic report generator
# ---------------------------------------------------------------------------

def run_diagnostic(args):
    """Run the complete diagnostic for the specified dataset and day."""
    verbose = args.verbose

    print("=" * 70)
    print("ORTHRUS / MSTC-PIDS Graph Day Diagnostic")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Day: {args.day}")
    print(f"Config: {args.config or 'default'}")
    print(f"Verbose: {verbose}")
    print("=" * 70)

    # Validate dataset
    if args.dataset not in DATASET_DEFAULT_CONFIG:
        print(f"ERROR: Unknown dataset '{args.dataset}'")
        print(f"Available datasets: {list(DATASET_DEFAULT_CONFIG.keys())}")
        return 1

    dataset_config = DATASET_DEFAULT_CONFIG[args.dataset]
    year_month = dataset_config["year_month"]
    start_day, end_day = dataset_config["start_end_day_range"]

    print(f"\nDataset config:")
    print(f"  year_month: {year_month}")
    print(f"  day range: {start_day} to {end_day}")
    print(f"  train_files: {dataset_config.get('train_files', [])}")
    print(f"  val_files: {dataset_config.get('val_files', [])}")
    print(f"  test_files: {dataset_config.get('test_files', [])}")

    # Build configuration
    print("\n[Initializing configuration...]")
    sys.argv = ["diagnose", args.dataset]
    if args.config:
        sys.argv.extend(["--config", args.config])

    try:
        runtime_args = get_runtime_required_args(args=["--config", args.config] if args.config else [])
        runtime_args.dataset = args.dataset
        cfg = get_yml_cfg(runtime_args)
    except Exception as e:
        print(f"ERROR: Failed to initialize configuration: {e}")
        return 2

    # Get supported relations from rel2id
    supported_relations = {v for k, v in rel2id.items() if isinstance(k, str)}
    print(f"\nSupported relations ({len(supported_relations)}): {sorted(supported_relations)}")

    # -------------------------------------------------------------------------
    # Phase A: Database Identity
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE A: Database Identity")
    print("=" * 70)

    cur = None
    connect = None

    try:
        cur, connect = init_database_connection(cfg)
        db_identity = get_db_identity(cur)
        print(f"current_database: {db_identity['current_database']}")
        print(f"version: {db_identity['version']}")
        print(f"server_version_num: {db_identity['server_version_num']}")
        print(f"port: {db_identity['port']}")
        print(f"session_timezone: {db_identity['session_timezone']}")
        print(f"data_directory: {db_identity['data_directory']}")

        # Check event_table
        exists, err = check_event_table_exists(cur)
        print(f"event_table exists: {exists}")
        if not exists:
            print("ERROR: event_table does not exist in database")
            return 1

        # -------------------------------------------------------------------------
        # Phase B: Event Table Global Stats
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE B: Event Table Global Stats")
        print("=" * 70)

        global_stats = get_event_table_global_stats(cur)
        print(f"total_rows: {global_stats.get('total_rows', 'N/A')}")
        if global_stats.get("min_timestamp_rec"):
            print(f"min_timestamp_rec (ns): {global_stats['min_timestamp_rec']}")
            print(f"min_timestamp_readable: {global_stats['min_timestamp_readable']}")
        if global_stats.get("max_timestamp_rec"):
            print(f"max_timestamp_rec (ns): {global_stats['max_timestamp_rec']}")
            print(f"max_timestamp_readable: {global_stats['max_timestamp_readable']}")
        if "error" in global_stats:
            print(f"WARNING: {global_stats['error']}")

        # -------------------------------------------------------------------------
        # Phase C: Adjacent Day Raw SQL Counts
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE C: Adjacent Day Raw SQL Counts")
        print("=" * 70)

        day_counts = {}
        for d in [args.day - 1, args.day, args.day + 1, args.day + 2]:
            if d < start_day or d >= end_day:
                continue
            count, start_ns, end_ns = count_day_raw_events(cur, year_month, d)
            day_counts[d] = {
                "count": count,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "start_readable": format_ns_timestamp(start_ns),
                "end_readable": format_ns_timestamp(end_ns),
            }
            marker = ""
            if d == args.day:
                marker = " <<< TARGET DAY"
            print(f"DAY {d}: {count:>10,} rows{marker}")
            print(f"  start: {day_counts[d]['start_readable']}")
            print(f"  end:   {day_counts[d]['end_readable']}")

        day2_count = day_counts.get(args.day, {}).get("count", 0)
        print(f"\nDAY 2 DIRECT RAW COUNT = {day2_count}")

        # -------------------------------------------------------------------------
        # Phase D: Day 2 Relation / Operation Distribution
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE D: Day 2 Relation / Operation Distribution")
        print("=" * 70)

        if day2_count == 0:
            print("Day 2 has NO raw events - skipping operation distribution.")
            supported_count = 0
            unsupported_relations = {}
        else:
            operation_dist = get_day_operation_distribution(cur, year_month, args.day)
            print(f"Operation distribution for day {args.day}:")
            for op, cnt in operation_dist:
                is_supported = op in supported_relations
                marker = "(SUPPORTED)" if is_supported else "(UNSUPPORTED)"
                print(f"  {op}: {cnt:>8,} {marker}")

            # Count supported vs unsupported
            supported_count = sum(cnt for op, cnt in operation_dist if op in supported_relations)
            unsupported_total = sum(cnt for op, cnt in operation_dist if op not in supported_relations)

            print(f"\nDay 2 raw: {day2_count}")
            print(f"Day 2 supported: {supported_count}")
            print(f"Day 2 unsupported: {unsupported_total}")

            # List unsupported operations if any
            if unsupported_total > 0:
                unsupported_relations = {
                    op: cnt for op, cnt in operation_dist if op not in supported_relations
                }
                print("\nUnsupported operations in day 2:")
                for op, cnt in sorted(unsupported_relations.items(), key=lambda x: -x[1]):
                    print(f"  {op}: {cnt}")
            else:
                unsupported_relations = {}

        print(f"\nDAY 2 SUPPORTED RELATION COUNT = {supported_count}")

        # -------------------------------------------------------------------------
        # Phase E: Production Streaming Comparison
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE E: Production Streaming Comparison")
        print("=" * 70)

        if day2_count == 0:
            print("Skipping streaming comparison (no raw events).")
            streaming_raw_count = 0
            streaming_filtered_count = 0
            direct_vs_streaming_raw_match = None
            supported_vs_filtered_match = None
        else:
            day2_start_ns = day_counts[args.day]["start_ns"]
            day2_end_ns = day_counts[args.day]["end_ns"]

            # Count using production stream_event_table
            streaming_raw_count = count_production_streaming_raw(cur, day2_start_ns, day2_end_ns)
            print(f"Production streaming raw: {streaming_raw_count}")

            # Count using production filtered_semantic_batches
            streaming_filtered_count = count_production_filtered_semantic(
                cur, day2_start_ns, day2_end_ns, supported_relations
            )
            print(f"Production semantic-filtered: {streaming_filtered_count}")

            # Compare
            direct_vs_streaming_raw_match = "PASS" if day2_count == streaming_raw_count else "MISMATCH"
            supported_vs_filtered_match = "PASS" if supported_count == streaming_filtered_count else "MISMATCH"

            print(f"\ndirect_raw == streaming_raw: {direct_vs_streaming_raw_match}")
            print(f"supported_relation == production_filtered: {supported_vs_filtered_match}")

        # -------------------------------------------------------------------------
        # Phase F: Sample Events
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE F: Sample Events (Day 2)")
        print("=" * 70)

        if day2_count == 0:
            print("No sample events (day 2 is empty).")
        else:
            samples = get_sample_events(cur, year_month, args.day, limit=5)
            if samples:
                print(f"Sample events (up to 5):")
                for i, event in enumerate(samples, 1):
                    src_node, src_idx, operation, dst_node, dst_idx, uuid, ts, eid = event
                    ts_readable = format_ns_timestamp(ts)
                    # Truncate long fields for display
                    src_display = str(src_node)[:40] + "..." if len(str(src_node)) > 40 else str(src_node)
                    dst_display = str(dst_node)[:40] + "..." if len(str(dst_node)) > 40 else str(dst_node)
                    uuid_display = str(uuid)[:20] + "..." if len(str(uuid)) > 20 else str(uuid)
                    print(f"\n  Event {i}:")
                    print(f"    src_node: {src_display}")
                    print(f"    src_index_id: {src_idx}")
                    print(f"    operation: {operation}")
                    print(f"    dst_node: {dst_display}")
                    print(f"    dst_index_id: {dst_idx}")
                    print(f"    timestamp: {ts_readable}")
            else:
                print("No sample events retrieved.")

        # -------------------------------------------------------------------------
        # Phase G: Diagnostic Summary
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PHASE G: Diagnostic Summary")
        print("=" * 70)

        # Determine case
        if day2_count == 0:
            # CASE A: Raw SQL is 0
            # Check if adjacent days have data
            adjacent_have_data = any(
                day_counts.get(d, {}).get("count", 0) > 0
                for d in [args.day - 1, args.day + 1]
                if d in day_counts
            )
            if adjacent_have_data:
                print("CASE_A_CANDIDATE:")
                print("  direct raw = 0, but adjacent days have data.")
                print("  This suggests day 2 is genuinely absent from the dump.")
            else:
                print("INCONCLUSIVE:")
                print("  No data in day 2 or adjacent days.")
                print("  Cannot determine if database has correct time range.")
        elif supported_count == 0:
            # CASE B: Raw > 0 but all relations unsupported
            print("CASE_B_CANDIDATE:")
            print(f"  direct raw = {day2_count}, but all relations are unsupported.")
            print("  Rel2id does not match any operation in day 2 data.")
            if unsupported_relations:
                print("  Unsupported operations found:")
                for op, cnt in sorted(unsupported_relations.items(), key=lambda x: -x[1])[:5]:
                    print(f"    {op}: {cnt}")
        else:
            # Some supported events exist
            print("BUILDABLE_EVENTS_PRESENT:")
            print(f"  production filtered semantic events = {streaming_filtered_count}")

            if direct_vs_streaming_raw_match == "MISMATCH":
                print("CASE_D_CANDIDATE:")
                print("  direct SQL and production streaming counts differ.")
                print("  This may indicate a streaming implementation issue.")
            elif supported_vs_filtered_match == "MISMATCH":
                print("CASE_D_CANDIDATE:")
                print("  supported relation count differs from production filtered count.")
                print("  This may indicate a filtering issue in production code.")
            else:
                print("INCONCLUSIVE:")
                print("  Day 2 has buildable events, but graph_2 is missing.")
                print("  The 'No events for day 2' log occurs BEFORE endpoint mapping.")
                print("  Further investigation of the graph building path is needed.")

        # Final note about the log
        print("\n" + "-" * 70)
        print("NOTE: The build log 'No events for day 2' occurs BEFORE endpoint")
        print("mapping / graph-window / graph-save. If BUILDABLE_EVENTS_PRESENT is")
        print("true, this suggests investigating the log path, code version, or")
        print("subsequent logic rather than the raw/semantic event filtering.")

        # -------------------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------------------
    finally:
        if cur is not None:
            cur.close()
        if connect is not None:
            connect.close()

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_diagnostic(args)


if __name__ == "__main__":
    sys.exit(main())
