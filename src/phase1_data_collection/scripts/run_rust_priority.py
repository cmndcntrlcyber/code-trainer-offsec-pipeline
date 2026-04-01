"""
phase1_data_collection/scripts/run_rust_priority.py

Capture Rust-containing repos first, then remaining repos.
Skips repos already captured on disk (by checking captures dir metadata).
"""
import argparse
import json
import logging
import os
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import load_config
from phase1_data_collection.scrapers.sqlite_catalog import SQLiteCatalog
from phase1_data_collection.scrapers.file_filter import OffSecFileFilter
from phase1_data_collection.scrapers.offsec_keywords import (
    classify_offsec_domain,
    extract_matched_keywords,
    detect_cve_references,
    get_mitre_tactics,
)
from phase1_data_collection.capture.vscode_automation import CaptureConfig
from phase1_data_collection.capture.screenshot_manager import ScreenshotManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("offsec_rust_priority.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_captured_repos_from_disk(captures_dir: str) -> set:
    """Scan capture metadata on disk to find already-captured repos."""
    repo_prefix = "data/offensive-security/repositories/"
    captured = set()
    for meta_path in glob.iglob(os.path.join(captures_dir, "*", "*", "metadata.json")):
        try:
            with open(meta_path) as f:
                content = f.read()
                if not content.strip():
                    continue
                d = json.loads(content)
                fp = d.get("file_path", "")
                if fp.startswith(repo_prefix):
                    parts = fp[len(repo_prefix):].split("/", 1)
                    if parts:
                        captured.add(parts[0])
        except Exception:
            pass
    return captured


def find_rust_repos(repos_dir: Path, repo_names: list) -> list:
    """Return repo names sorted by Rust file count (descending)."""
    rust_counts = {}
    for repo_name in repo_names:
        repo_path = repos_dir / repo_name
        if not repo_path.is_dir():
            continue
        count = 0
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {
                ".git", "node_modules", "target", "vendor", "__pycache__", ".venv"
            }]
            for f in files:
                if f.endswith(".rs"):
                    count += 1
        if count > 0:
            rust_counts[repo_name] = count

    sorted_repos = sorted(rust_counts.keys(), key=lambda r: rust_counts[r], reverse=True)
    logger.info(f"Found {len(sorted_repos)} repos with {sum(rust_counts.values())} Rust files")
    for r in sorted_repos[:10]:
        logger.info(f"  {r}: {rust_counts[r]} .rs files")
    return sorted_repos


def process_repo(repo_path, repo_name, screenshot_mgr, catalog):
    """Capture files for a single repo and enrich with offsec metadata."""
    full_name = repo_name.replace("_", "/", 1)
    files = list(OffSecFileFilter.get_code_files(repo_path))

    if not files:
        return 0

    logger.info(f"Processing {repo_name}: {len(files)} files")
    results = screenshot_mgr.capture_files(files, full_name)

    for result in results:
        if not result.success:
            continue
        capture_id = catalog.get_capture_id(result.file_hash)
        if not capture_id:
            continue
        source_text = result.source_code[:5000]
        domain = classify_offsec_domain(
            description=str(result.file_path), readme_text=source_text,
        )
        keywords = extract_matched_keywords(
            description=str(result.file_path), readme_text=source_text,
        )
        cve_ids = detect_cve_references(source_text)
        mitre_tactics = get_mitre_tactics(domain)
        catalog.add_offsec_metadata(
            capture_id=capture_id, domain=domain, mitre_tactics=mitre_tactics,
            keywords_matched=keywords, has_cve=bool(cve_ids), cve_ids=cve_ids,
        )

    return sum(1 for r in results if r.success)


def main():
    parser = argparse.ArgumentParser(
        description="Rust-Priority Offensive Security Capture"
    )
    parser.add_argument("--config", required=True, help="Path to offsec_config.yaml")
    parser.add_argument("--rust-only", action="store_true",
                        help="Only capture Rust-containing repos, skip the rest")
    args = parser.parse_args()

    config = load_config(args.config)
    dc = config["data_collection"]

    catalog = SQLiteCatalog(dc["catalog_db"])
    catalog.init_offsec_schema()

    repos_dir = Path(dc["repos_dir"])
    captures_dir = Path(dc["captures_dir"])

    # Build skip set from disk (not DB, since DB is empty)
    logger.info("Scanning existing captures on disk...")
    captured_repos = get_captured_repos_from_disk(str(captures_dir))
    logger.info(f"Found {len(captured_repos)} already-captured repos on disk")

    # Get remaining repos
    all_repos = [p.name for p in repos_dir.iterdir() if p.is_dir()]
    remaining = [r for r in all_repos if r not in captured_repos]
    logger.info(f"Remaining repos to process: {len(remaining)}")

    # Find and prioritize Rust repos
    logger.info("Scanning for Rust files in remaining repos...")
    rust_repos = find_rust_repos(repos_dir, remaining)
    non_rust = [r for r in remaining if r not in set(rust_repos)]

    if args.rust_only:
        ordered = rust_repos
        logger.info(f"Rust-only mode: processing {len(ordered)} repos")
    else:
        ordered = rust_repos + non_rust
        logger.info(f"Processing {len(rust_repos)} Rust repos first, then {len(non_rust)} others")

    # Setup capture
    capture_config = CaptureConfig(
        viewport_width=dc.get("viewport_width", 2560),
        viewport_height=dc.get("viewport_height", 1440),
        font_size=dc.get("font_size", 14),
        theme=dc.get("theme", "Default Dark+"),
    )

    screenshot_mgr = ScreenshotManager(
        catalog=catalog,
        output_dir=captures_dir,
        config=capture_config,
        num_workers=dc.get("capture_workers", 8),
        rotate_themes=True,
    )

    total_captured = 0
    for i, repo_name in enumerate(ordered):
        repo_path = repos_dir / repo_name
        captured = process_repo(repo_path, repo_name, screenshot_mgr, catalog)
        total_captured += captured
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(ordered)} repos, {total_captured} total captures")

    stats = catalog.get_statistics()
    logger.info(f"Complete: {total_captured} new captures across {len(ordered)} repos")
    logger.info(f"Catalog: {stats}")


if __name__ == "__main__":
    main()
