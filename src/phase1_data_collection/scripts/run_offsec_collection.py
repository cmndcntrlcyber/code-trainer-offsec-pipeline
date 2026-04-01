"""
phase1_data_collection/scripts/run_offsec_collection.py

Orchestrator for offensive security specialized data collection.
Flow: discover offsec repos -> filter files -> parallel capture -> enrich metadata
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import load_config
from phase1_data_collection.scrapers.sqlite_catalog import SQLiteCatalog
from phase1_data_collection.scrapers.offsec_scraper import OffSecScraper
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
        logging.FileHandler("offsec_collection.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Offensive Security Data Collection Pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to offsec_config.yaml")
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip GitHub scraping, use existing cloned repos",
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Skip screenshot capture, only scrape repos",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dc = config["data_collection"]

    # Initialize catalog with offsec schema
    catalog = SQLiteCatalog(dc["catalog_db"])
    catalog.init_offsec_schema()
    logger.info(f"Catalog initialized: {dc['catalog_db']}")

    repos_dir = Path(dc["repos_dir"])
    captures_dir = Path(dc["captures_dir"])
    repos_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1A: Discover and clone offensive security repositories
    repo_paths = []
    if not args.skip_scraping:
        scraper = OffSecScraper(
            token=dc["github_token"],
            output_dir=repos_dir,
            catalog=catalog,
            seed_orgs=dc.get("seed_orgs", []),
            seed_users=dc.get("seed_users", []),
            topics=dc.get("topics", []),
            search_queries=dc.get("search_queries", []),
            languages=dc.get("languages", []),
            min_quality_score=dc.get("min_quality_score", 20),
        )

        for repo_path in scraper.collect_all():
            repo_paths.append(repo_path)

        logger.info(f"Cloned {len(repo_paths)} offensive security repositories")
    else:
        if repos_dir.exists():
            repo_paths = [p for p in repos_dir.iterdir() if p.is_dir()]
            logger.info(f"Using {len(repo_paths)} existing repositories")
        else:
            logger.error(f"Repos directory not found: {repos_dir}")
            return

    if args.skip_capture:
        logger.info("Skipping capture phase (--skip-capture)")
        stats = catalog.get_statistics()
        logger.info(f"Catalog stats: {stats}")
        return

    # Phase 1B: Filter files and capture screenshots
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

    total_files = 0
    total_captured = 0

    # Build set of already-captured repos to skip on resume
    import sqlite3
    with sqlite3.connect(dc["catalog_db"]) as conn:
        captured_repo_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT r.full_name FROM captures c "
                "JOIN repositories r ON c.repo_id = r.id"
            ).fetchall()
        )
    logger.info(f"Skipping {len(captured_repo_ids)} already-captured repos")

    for repo_path in repo_paths:
        repo_name = repo_path.name
        # Convert dir name (owner_repo) to DB format (owner/repo)
        full_name = repo_name.replace("_", "/", 1)
        if full_name in captured_repo_ids:
            continue

        files = list(OffSecFileFilter.get_code_files(repo_path))

        if not files:
            logger.debug(f"No valid files in {repo_name}")
            continue

        total_files += len(files)
        logger.info(f"Processing {repo_name}: {len(files)} files")

        results = screenshot_mgr.capture_files(files, full_name)

        # Enrich with offsec metadata
        for result in results:
            if not result.success:
                continue

            capture_id = catalog.get_capture_id(result.file_hash)
            if not capture_id:
                continue

            # Classify from source code + file path
            source_text = result.source_code[:5000]
            domain = classify_offsec_domain(
                description=str(result.file_path),
                readme_text=source_text,
            )
            keywords = extract_matched_keywords(
                description=str(result.file_path),
                readme_text=source_text,
            )
            cve_ids = detect_cve_references(source_text)
            mitre_tactics = get_mitre_tactics(domain)

            catalog.add_offsec_metadata(
                capture_id=capture_id,
                domain=domain,
                mitre_tactics=mitre_tactics,
                keywords_matched=keywords,
                has_cve=bool(cve_ids),
                cve_ids=cve_ids,
            )

        captured = sum(1 for r in results if r.success)
        total_captured += captured

    # Final stats
    stats = catalog.get_statistics()
    logger.info(f"Collection complete: {total_captured}/{total_files} files captured")
    logger.info(f"Catalog: {stats['total_repos']} repos, {stats['total_captures']} captures")
    logger.info(f"By language: {stats['by_language']}")
    logger.info(f"By category: {stats['by_category']}")
    logger.info(f"Avg quality: {stats['avg_quality']:.1f}")


if __name__ == "__main__":
    main()
