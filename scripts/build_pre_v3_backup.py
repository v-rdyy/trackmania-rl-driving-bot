"""Build a checksum-pinned source and V2-evidence backup before V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "artifacts" / "backups"
EVIDENCE_PATHS = (
    WORKSPACE_ROOT / "checkpoints" / "reward_v2",
    WORKSPACE_ROOT / "runs" / "reward_v2",
    WORKSPACE_ROOT / "tensorboard" / "reward_v2_dense_speed_1",
    WORKSPACE_ROOT / "artifacts" / "replays" / "progression",
    WORKSPACE_ROOT / "artifacts" / "smoke" / "gas_direction_probe_corrected.json",
    WORKSPACE_ROOT / "artifacts" / "telemetry" / "gas_direction_probe_corrected.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive Git history and V2 evidence before reward V3."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_evidence_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(f"backup evidence path is missing: {path}")
    return sorted(set(files), key=lambda candidate: str(candidate).casefold())


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={WORKSPACE_ROOT.as_posix()}",
            *arguments,
        ],
        cwd=WORKSPACE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    tracked_status = git_output("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError(
            "refusing pre-V3 backup with uncommitted tracked changes:\n"
            f"{tracked_status}"
        )
    commit = git_output("rev-parse", "HEAD")
    short_commit = commit[:7]
    evidence_files = collect_evidence_files(EVIDENCE_PATHS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / f"trackmania_v2_pre_v3_{short_commit}.zip"
    checksum_path = archive_path.with_suffix(".zip.sha256")
    if archive_path.exists() or checksum_path.exists():
        raise FileExistsError(f"refusing to replace existing backup: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="trackmania-v2-backup-") as temp_dir:
        source_bundle = Path(temp_dir) / "trackmania.git.bundle"
        git_output("bundle", "create", str(source_bundle), "--all")
        source_bundle_hash = sha256(source_bundle)
        evidence_manifest = [
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in evidence_files
        ]
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "checksum-pinned V2 evidence backup before any V3 training",
            "source_commit": commit,
            "source_bundle": {
                "archive_path": "source/trackmania.git.bundle",
                "bytes": source_bundle.stat().st_size,
                "sha256": source_bundle_hash,
            },
            "evidence_file_count": len(evidence_manifest),
            "evidence_bytes": sum(item["bytes"] for item in evidence_manifest),
            "evidence_files": evidence_manifest,
        }
        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(source_bundle, "source/trackmania.git.bundle")
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            )
            for path in evidence_files:
                relative = str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
                archive.write(path, f"evidence/{relative}")

    archive_hash = sha256(archive_path)
    checksum_path.write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="utf-8",
    )
    print(f"pre-V3 backup: {archive_path}")
    print(f"evidence files: {len(evidence_files)}")
    print(f"archive bytes: {archive_path.stat().st_size}")
    print(f"SHA-256: {archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
