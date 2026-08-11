"""Google Colab production helpers for PostgreSQL archive restoration."""

from __future__ import annotations

import glob
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path


PGDG_MAJOR = "18"
REQUIRED_TABLES = {
    "event_table",
    "subject_node_table",
    "file_node_table",
    "netflow_node_table",
}


def run_cmd(command, *, env=None, input_text=None, display_command=None, check=True):
    """Run a command without ever logging hidden command arguments or stdin."""
    shown = display_command if display_command is not None else command
    if isinstance(shown, (list, tuple)):
        shown = " ".join(str(part) for part in shown)
    print(f"$ {shown}")
    return subprocess.run(
        [str(part) for part in command],
        env=env,
        input=input_text,
        text=True,
        check=check,
        capture_output=False,
    )


def _capture(command, *, env=None, check=True):
    return subprocess.run(
        [str(part) for part in command],
        env=env,
        text=True,
        check=check,
        capture_output=True,
    )


def pg_restore_candidates():
    """Return pg_restore candidates ordered from newest major to oldest.

    The symlink at ``/usr/bin/pg_restore`` is intentionally resolved so
    downstream code can derive the actual PostgreSQL major from the
    binary path, rather than guessing it from the symlink directory
    (``bin``/``usr``).
    """
    candidates = []
    seen = set()
    for path in sorted(
        glob.glob("/usr/lib/postgresql/*/bin/pg_restore"),
        reverse=True,
    ):
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    current = shutil.which("pg_restore")
    if current:
        resolved = str(Path(current).resolve())
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


_PG_VERSION_RE = re.compile(r"\(PostgreSQL\)\s+(\d+)")


def parse_pg_restore_major(version_text):
    """Extract the PostgreSQL major from a ``pg_restore --version`` line.

    Returns the major as a string (e.g. ``"18"``), or ``None`` if the
    text does not look like a stock PostgreSQL version banner. The
    parser deliberately does *not* try to guess from the path layout.
    """
    if not version_text:
        return None
    match = _PG_VERSION_RE.search(version_text)
    return match.group(1) if match else None


def archive_is_readable(pg_restore, dump_path):
    result = _capture([pg_restore, "--list", dump_path], check=False)
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def choose_compatible_pg_restore(dump_path, candidates=None):
    """Return the first client that can parse the archive, else None."""
    for candidate in candidates or pg_restore_candidates():
        ok, reason = archive_is_readable(candidate, dump_path)
        version = _capture([candidate, "--version"], check=False).stdout.strip()
        major = parse_pg_restore_major(version) or "?"
        print(
            f"Archive check: {candidate} ({version}) -> "
            f"{'ok' if ok else 'unsupported'} (major={major})"
        )
        if ok:
            return candidate
        if reason:
            print(reason.splitlines()[-1])
    return None


def _os_codename():
    values = {}
    with open("/etc/os-release", "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" in line:
                key, value = line.rstrip().split("=", 1)
                values[key] = value.strip('"')
    return values.get("VERSION_CODENAME", "jammy")


def install_pgdg_client_server(major=PGDG_MAJOR):
    """Idempotently install the validated PGDG fallback."""
    run_cmd(["apt-get", "update", "-qq"])
    run_cmd([
        "apt-get", "install", "-qq", "-y",
        "ca-certificates", "curl", "gnupg", "postgresql-common",
    ])
    key_path = Path("/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc")
    run_cmd([
        "curl", "-fsSL", "https://www.postgresql.org/media/keys/ACCC4CF8.asc",
        "-o", key_path,
    ])
    source_path = Path("/etc/apt/sources.list.d/pgdg.list")
    source_path.write_text(
        f"deb [signed-by={key_path}] https://apt.postgresql.org/pub/repos/apt "
        f"{_os_codename()}-pgdg main\n",
        encoding="utf-8",
    )
    run_cmd(["apt-get", "update", "-qq"])
    run_cmd([
        "apt-get", "install", "-qq", "-y",
        f"postgresql-{major}", f"postgresql-client-{major}",
    ])


def ensure_compatible_postgres(dump_path):
    dump_path = Path(dump_path)
    if not dump_path.is_file():
        raise FileNotFoundError(f"Database dump not found: {dump_path}")
    print(f"Dump: {dump_path} ({dump_path.stat().st_size / 1024**3:.3f} GB)")

    pg_restore = choose_compatible_pg_restore(dump_path)
    if pg_restore is None:
        print(f"No installed pg_restore can read this archive; installing PGDG {PGDG_MAJOR}.")
        install_pgdg_client_server(PGDG_MAJOR)
        pg_restore = choose_compatible_pg_restore(dump_path)
    if pg_restore is None:
        raise RuntimeError("No pg_restore client can read the archive after PGDG fallback")

    # Resolve any symlink (notably /usr/bin/pg_restore on Debian/Ubuntu)
    # so the major is derived from the real binary location, not from the
    # ``bin``/``usr`` directory of the symlink.
    resolved_pg_restore = Path(pg_restore).resolve()
    pg_bin = resolved_pg_restore.parent
    psql = pg_bin / "psql"
    if not psql.is_file():
        raise RuntimeError(f"Matching psql is missing: {psql}")

    version_text = _capture(
        [str(resolved_pg_restore), "--version"], check=False
    ).stdout.strip()
    major = parse_pg_restore_major(version_text)
    if not major or not major.isdigit():
        raise RuntimeError(
            "Could not determine PostgreSQL major from "
            f"{resolved_pg_restore}: version output was {version_text!r}"
        )
    return str(resolved_pg_restore), str(psql), major


def _clusters():
    result = _capture(["pg_lsclusters", "--no-header"], check=False)
    clusters = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            clusters.append({
                "major": fields[0],
                "name": fields[1],
                "port": fields[2],
                "status": fields[3],
            })
    return clusters


def ensure_cluster(target_major, target_name="main", port="5432"):
    """Move the compatible cluster to 5432 without deleting any cluster data."""
    clusters = _clusters()
    target = next(
        (item for item in clusters
         if item["major"] == str(target_major) and item["name"] == target_name),
        None,
    )
    if target is None:
        run_cmd(["pg_createcluster", str(target_major), target_name])
        clusters = _clusters()

    for cluster in clusters:
        is_target = (
            cluster["major"] == str(target_major)
            and cluster["name"] == target_name
        )
        target_needs_restart = (
            is_target and cluster["status"] == "online"
            and cluster["port"] != str(port)
        )
        port_conflict = cluster["port"] == str(port) and not is_target
        if cluster["status"] == "online" and (target_needs_restart or port_conflict):
            run_cmd(["pg_ctlcluster", cluster["major"], cluster["name"], "stop"])

    run_cmd([
        "pg_conftool", str(target_major), target_name, "set", "port", str(port)
    ])
    run_cmd(["pg_ctlcluster", str(target_major), target_name, "start"], check=False)
    ready = False
    for _ in range(30):
        if _capture(["pg_isready", "-h", "localhost", "-p", str(port)], check=False).returncode == 0:
            ready = True
            break
        time.sleep(1)
    if not ready:
        raise RuntimeError(f"PostgreSQL {target_major}/{target_name} is not ready")
    run_cmd(["pg_lsclusters"])


def _safe_identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe PostgreSQL identifier: {value!r}")
    return value


def set_postgres_password(psql, password):
    """Set the first password over the Unix socket; password stays in stdin."""
    sql = f"ALTER USER postgres WITH PASSWORD '{password.replace(chr(39), chr(39)*2)}';\n"
    run_cmd(
        ["runuser", "-u", "postgres", "--", psql, "-v", "ON_ERROR_STOP=1"],
        input_text=sql,
        display_command=f"runuser -u postgres -- {psql} [REDACTED SQL]",
    )


def _pg_env(password):
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    return env


def database_has_required_tables(psql, db_name, password):
    query = (
        "SELECT tablename FROM pg_tables WHERE schemaname='public' "
        "ORDER BY tablename;"
    )
    result = _capture([
        psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
        "-d", db_name, "-Atc", query,
    ], env=_pg_env(password), check=False)
    tables = set(result.stdout.splitlines())
    return REQUIRED_TABLES.issubset(tables), tables


def restore_database(dump_path, db_name, *, force=False):
    """Restore a custom archive idempotently after compatibility validation."""
    db_name = _safe_identifier(db_name)
    pg_restore, psql, major = ensure_compatible_postgres(dump_path)
    ensure_cluster(major)

    password = secrets.token_urlsafe(24)
    set_postgres_password(psql, password)
    env = _pg_env(password)
    run_cmd([psql, "--version"])
    run_cmd([
        psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
        "-Atc", "SELECT version();",
    ], env=env, display_command=f"{psql} [authenticated server version query]")

    exists = _capture([
        psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
        "-Atc", f"SELECT 1 FROM pg_database WHERE datname='{db_name}';",
    ], env=env).stdout.strip() == "1"

    complete = False
    if exists:
        complete, tables = database_has_required_tables(psql, db_name, password)
        print(f"Existing database tables: {sorted(tables)}")
    if exists and complete and not force:
        print(f"Database {db_name} is already complete; restore skipped.")
    else:
        if exists and force:
            run_cmd([
                psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
                "-c", f"DROP DATABASE {db_name};",
            ], env=env, display_command=f"{psql} [drop requested database]")
            exists = False
        elif exists and not complete:
            raise RuntimeError(
                f"Database {db_name} exists but is incomplete; set "
                "FORCE_DATABASE_RESTORE=True to replace it."
            )
        if not exists:
            run_cmd([
                psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
                "-c", f"CREATE DATABASE {db_name};",
            ], env=env, display_command=f"{psql} [create requested database]")
        # This check is deliberately repeated immediately before restore.
        ok, reason = archive_is_readable(pg_restore, dump_path)
        if not ok:
            raise RuntimeError(f"Archive compatibility changed: {reason}")
        run_cmd([
            pg_restore, "--no-owner", "--exit-on-error",
            "-h", "localhost", "-p", "5432", "-U", "postgres",
            "-d", db_name, dump_path,
        ], env=env, display_command=f"{pg_restore} [restore archive into {db_name}]")

    complete, tables = database_has_required_tables(psql, db_name, password)
    if not complete:
        raise RuntimeError(
            f"Restored database lacks required tables: {sorted(REQUIRED_TABLES - tables)}"
        )
    run_cmd([
        psql, "-h", "localhost", "-p", "5432", "-U", "postgres",
        "-d", db_name, "-Atc", "SELECT 1 FROM event_table LIMIT 1;",
    ], env=env, display_command=f"{psql} [event_table quick read]")
    return {
        "host": "localhost",
        "port": "5432",
        "user": "postgres",
        "password": password,
        "pg_restore": pg_restore,
        "psql": psql,
        "major": major,
    }
