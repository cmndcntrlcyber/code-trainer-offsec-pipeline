"""
phase1_data_collection/scrapers/sqlite_catalog.py

SQLite-based repository and capture catalog.
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional


class SQLiteCatalog:
    """SQLite-based repository and capture catalog."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY,
            full_name TEXT UNIQUE NOT NULL,
            clone_url TEXT NOT NULL,
            language TEXT,
            stars INTEGER DEFAULT 0,
            forks INTEGER DEFAULT 0,
            quality_score REAL DEFAULT 0,
            category TEXT DEFAULT 'general',
            size_kb INTEGER DEFAULT 0,
            cloned_at TIMESTAMP,
            local_path TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY,
            repo_id INTEGER REFERENCES repositories(id),
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            language TEXT,
            line_count INTEGER,
            screenshot_count INTEGER DEFAULT 0,
            quality_score REAL DEFAULT 0,
            processed BOOLEAN DEFAULT FALSE,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata_json TEXT,
            UNIQUE(repo_id, file_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_repos_quality ON repositories(quality_score DESC);
        CREATE INDEX IF NOT EXISTS idx_repos_language ON repositories(language);
        CREATE INDEX IF NOT EXISTS idx_repos_category ON repositories(category);
        CREATE INDEX IF NOT EXISTS idx_captures_processed ON captures(processed);
        CREATE INDEX IF NOT EXISTS idx_captures_language ON captures(language);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def add_repository(self, repo, local_path: Optional[Path] = None):
        """Add or update repository."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO repositories (
                    full_name, clone_url, language, stars, forks,
                    quality_score, category, size_kb, local_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(full_name) DO UPDATE SET
                    stars = excluded.stars,
                    forks = excluded.forks,
                    quality_score = excluded.quality_score,
                    local_path = excluded.local_path
            """, (
                repo.full_name, repo.clone_url, repo.language,
                repo.stars, repo.forks, repo.quality_score, repo.category,
                repo.size_kb, str(local_path) if local_path else None,
                json.dumps({k: v for k, v in vars(repo).items() if not k.startswith('_')})
            ))

    def add_capture(
        self,
        repo_name: str,
        file_path: Path,
        file_hash: str,
        language: str,
        line_count: int,
        screenshot_count: int,
        metadata: dict
    ):
        """Add capture record."""
        with sqlite3.connect(self.db_path) as conn:
            repo_id = conn.execute(
                "SELECT id FROM repositories WHERE full_name = ?",
                (repo_name,)
            ).fetchone()

            if repo_id:
                conn.execute("""
                    INSERT INTO captures (
                        repo_id, file_path, file_hash, language,
                        line_count, screenshot_count, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id, file_hash) DO UPDATE SET
                        screenshot_count = excluded.screenshot_count
                """, (
                    repo_id[0], str(file_path), file_hash, language,
                    line_count, screenshot_count, json.dumps(metadata)
                ))

    def init_offsec_schema(self):
        """Add offensive security metadata table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS offsec_metadata (
                    id INTEGER PRIMARY KEY,
                    capture_id INTEGER REFERENCES captures(id),
                    domain TEXT NOT NULL,
                    mitre_tactics TEXT,
                    keywords_matched TEXT,
                    has_cve_reference BOOLEAN DEFAULT FALSE,
                    cve_ids TEXT,
                    UNIQUE(capture_id)
                );
                CREATE INDEX IF NOT EXISTS idx_offsec_domain ON offsec_metadata(domain);
            """)

    def add_offsec_metadata(
        self,
        capture_id: int,
        domain: str,
        mitre_tactics: list = None,
        keywords_matched: list = None,
        has_cve: bool = False,
        cve_ids: list = None
    ):
        """Add offensive security metadata for a capture."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO offsec_metadata (
                    capture_id, domain, mitre_tactics, keywords_matched,
                    has_cve_reference, cve_ids
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(capture_id) DO UPDATE SET
                    domain = excluded.domain,
                    mitre_tactics = excluded.mitre_tactics,
                    keywords_matched = excluded.keywords_matched,
                    has_cve_reference = excluded.has_cve_reference,
                    cve_ids = excluded.cve_ids
            """, (
                capture_id, domain,
                ",".join(mitre_tactics) if mitre_tactics else None,
                json.dumps(keywords_matched) if keywords_matched else None,
                has_cve,
                ",".join(cve_ids) if cve_ids else None
            ))

    def get_capture_id(self, file_hash: str) -> Optional[int]:
        """Get capture ID by file hash."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM captures WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return row[0] if row else None

    def get_statistics(self) -> Dict:
        """Get catalog statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {
                "total_repos": conn.execute("SELECT COUNT(*) FROM repositories").fetchone()[0],
                "total_captures": conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0],
                "processed_captures": conn.execute(
                    "SELECT COUNT(*) FROM captures WHERE processed = TRUE"
                ).fetchone()[0],
                "avg_quality": conn.execute(
                    "SELECT AVG(quality_score) FROM repositories"
                ).fetchone()[0] or 0,
                "by_language": dict(conn.execute(
                    "SELECT language, COUNT(*) FROM repositories GROUP BY language"
                ).fetchall()),
                "by_category": dict(conn.execute(
                    "SELECT category, COUNT(*) FROM repositories GROUP BY category"
                ).fetchall())
            }
            return stats
