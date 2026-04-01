"""
phase1_data_collection/scrapers/github_scraper.py

GitHub repository discovery with quality scoring and SQLite catalog.
"""
import logging
from pathlib import Path
from typing import List, Optional, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from git import Repo

from .quality_scorer import QualityScorer
from .sqlite_catalog import SQLiteCatalog

logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """Repository metadata with quality scoring."""
    full_name: str
    clone_url: str
    stars: int
    forks: int
    language: str
    size_kb: int
    default_branch: str
    description: str = ""
    topics: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    open_issues: int = 0
    has_readme: bool = True
    license: str = ""
    quality_score: float = 0.0
    category: str = "general"


class GitHubScraper:
    """GitHub repository discovery with quality scoring."""

    GITHUB_API = "https://api.github.com"

    def __init__(
        self,
        token: str,
        output_dir: Path,
        catalog: SQLiteCatalog,
        languages: List[str] = None,
        min_stars: int = 10,
        min_quality_score: float = 30.0,
        max_size_kb: int = 100_000
    ):
        self.token = token
        self.output_dir = Path(output_dir)
        self.catalog = catalog
        self.languages = languages or [
            "Python", "JavaScript", "TypeScript", "Java",
            "C++", "Go", "Rust", "C#"
        ]
        self.min_stars = min_stars
        self.min_quality_score = min_quality_score
        self.max_size_kb = max_size_kb
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_repo_details(self, repo_data: dict) -> RepoMetadata:
        """Convert API response to RepoMetadata."""
        repo = RepoMetadata(
            full_name=repo_data["full_name"],
            clone_url=repo_data["clone_url"],
            stars=repo_data["stargazers_count"],
            forks=repo_data.get("forks_count", 0),
            language=repo_data.get("language", "Unknown"),
            size_kb=repo_data["size"],
            default_branch=repo_data.get("default_branch", "main"),
            description=repo_data.get("description", "") or "",
            topics=repo_data.get("topics", []),
            created_at=repo_data.get("created_at", ""),
            updated_at=repo_data.get("updated_at", ""),
            open_issues=repo_data.get("open_issues_count", 0),
            license=repo_data.get("license", {}).get("spdx_id", "") if repo_data.get("license") else ""
        )

        repo.quality_score = QualityScorer.score_repository(repo)
        repo.category = QualityScorer.classify_category(repo)

        return repo

    def search_repositories(
        self,
        language: str,
        page: int = 1,
        per_page: int = 100
    ) -> List[RepoMetadata]:
        """Search GitHub for repositories."""
        query = f"language:{language} stars:>={self.min_stars} size:<{self.max_size_kb}"
        url = f"{self.GITHUB_API}/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "page": page,
            "per_page": per_page
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()

        repos = []
        for item in data.get("items", []):
            repo = self._fetch_repo_details(item)
            if repo.quality_score >= self.min_quality_score:
                repos.append(repo)

        logger.info(f"Found {len(repos)} quality {language} repos (page {page})")
        return repos

    def clone_repository(self, repo: RepoMetadata) -> Optional[Path]:
        """Clone a single repository."""
        repo_path = self.output_dir / repo.full_name.replace("/", "_")

        if repo_path.exists():
            logger.debug(f"Repository exists: {repo.full_name}")
            return repo_path

        try:
            Repo.clone_from(
                repo.clone_url,
                repo_path,
                depth=1,
                branch=repo.default_branch
            )
            logger.info(f"Cloned: {repo.full_name} (score: {repo.quality_score})")
            self.catalog.add_repository(repo, repo_path)
            return repo_path
        except Exception as e:
            logger.error(f"Clone failed {repo.full_name}: {e}")
            return None

    def collect_repositories(
        self,
        repos_per_language: int = 500,
        max_workers: int = 4
    ) -> Generator[Path, None, None]:
        """Collect high-quality repositories."""
        all_repos = []

        for language in self.languages:
            pages_needed = (repos_per_language + 99) // 100
            language_repos = []

            for page in range(1, pages_needed + 1):
                repos = self.search_repositories(language, page)
                language_repos.extend(repos)
                if len(language_repos) >= repos_per_language or len(repos) < 100:
                    break

            language_repos.sort(key=lambda r: r.quality_score, reverse=True)
            all_repos.extend(language_repos[:repos_per_language])

        logger.info(f"Discovered {len(all_repos)} quality repositories")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.clone_repository, repo): repo for repo in all_repos}
            for future in as_completed(futures):
                repo_path = future.result()
                if repo_path:
                    yield repo_path
