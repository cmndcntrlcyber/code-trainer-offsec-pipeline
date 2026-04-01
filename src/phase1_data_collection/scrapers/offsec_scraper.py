"""
phase1_data_collection/scrapers/offsec_scraper.py

Offensive security repository discovery with domain-aware quality scoring.
Discovers repos via org enumeration, topic search, and keyword queries.
"""
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Generator, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from git import Repo

from .github_scraper import RepoMetadata
from .sqlite_catalog import SQLiteCatalog
from .offsec_keywords import (
    classify_offsec_domain,
    extract_matched_keywords,
    detect_cve_references,
    get_mitre_tactics,
    OFFSEC_KEYWORDS,
)

logger = logging.getLogger(__name__)


class OffSecScraper:
    """Offensive security focused repository discovery and collection."""

    GITHUB_API = "https://api.github.com"
    # GitHub search API: 30 requests/min for authenticated users
    SEARCH_DELAY = 2.5  # seconds between search requests

    def __init__(
        self,
        token: str,
        output_dir: Path,
        catalog: SQLiteCatalog,
        seed_orgs: List[str] = None,
        seed_users: List[str] = None,
        topics: List[str] = None,
        search_queries: List[str] = None,
        languages: List[str] = None,
        min_quality_score: float = 20.0,
    ):
        self.token = token
        self.output_dir = Path(output_dir)
        self.catalog = catalog
        self.seed_orgs = seed_orgs or []
        self.seed_users = seed_users or []
        self.topics = topics or []
        self.search_queries = search_queries or []
        self.languages = languages or []
        self.min_quality_score = min_quality_score
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seen: Set[str] = set()

    def _api_get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make a GitHub API GET request with error handling."""
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code == 403:
                # Rate limited — check reset time
                reset = int(response.headers.get("X-RateLimit-Reset", 0))
                wait = max(0, reset - int(time.time())) + 1
                logger.warning(f"Rate limited, waiting {wait}s")
                time.sleep(min(wait, 120))
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} — {e}")
            return None

    def _to_repo_metadata(self, repo_data: dict) -> Optional[RepoMetadata]:
        """Convert GitHub API response to RepoMetadata with offsec scoring."""
        try:
            full_name = repo_data["full_name"]
            if full_name in self._seen:
                return None
            self._seen.add(full_name)

            repo = RepoMetadata(
                full_name=full_name,
                clone_url=repo_data["clone_url"],
                stars=repo_data.get("stargazers_count", 0),
                forks=repo_data.get("forks_count", 0),
                language=repo_data.get("language", "Unknown") or "Unknown",
                size_kb=repo_data.get("size", 0),
                default_branch=repo_data.get("default_branch", "main"),
                description=repo_data.get("description", "") or "",
                topics=repo_data.get("topics", []),
                created_at=repo_data.get("created_at", ""),
                updated_at=repo_data.get("updated_at", ""),
                open_issues=repo_data.get("open_issues_count", 0),
                license=(
                    repo_data.get("license", {}).get("spdx_id", "")
                    if repo_data.get("license")
                    else ""
                ),
            )

            repo.quality_score = self.score_offsec_quality(repo)
            repo.category = classify_offsec_domain(repo.description, repo.topics)

            return repo
        except (KeyError, TypeError) as e:
            logger.debug(f"Skipping malformed repo data: {e}")
            return None

    def score_offsec_quality(self, repo: RepoMetadata) -> float:
        """Security-specific quality scoring (0-100).

        Components:
          Relevance    30 pts — keyword density, CVE references, MITRE TIDs
          Code Quality 20 pts — structured project, has requirements/setup
          Recency      20 pts — updated within 1 year
          Community    15 pts — stars, forks (log scale)
          Completeness 15 pts — README, examples, documentation
        """
        scores: Dict[str, float] = {}

        # Relevance (30 pts)
        text = f"{repo.description} {' '.join(repo.topics)}".lower()
        matched = extract_matched_keywords(repo.description, repo.topics)
        keyword_score = min(20, len(matched) * 2)
        cve_bonus = 5 if detect_cve_references(text) else 0
        mitre_bonus = 5 if any(
            t in text for t in ["mitre", "att&ck", "ta00", "t1"]
        ) else 0
        scores["relevance"] = min(30, keyword_score + cve_bonus + mitre_bonus)

        # Code Quality (20 pts)
        quality = 10  # Base
        if repo.size_kb > 10:
            quality += 5  # Not just a single script
        if repo.license:
            quality += 5
        scores["code_quality"] = min(20, quality)

        # Recency (20 pts)
        try:
            updated = datetime.fromisoformat(repo.updated_at.replace("Z", "+00:00"))
            days_since = (datetime.now(updated.tzinfo) - updated).days
            if days_since < 90:
                scores["recency"] = 20
            elif days_since < 180:
                scores["recency"] = 15
            elif days_since < 365:
                scores["recency"] = 10
            elif days_since < 730:
                scores["recency"] = 5
            else:
                scores["recency"] = 0
        except (ValueError, TypeError):
            scores["recency"] = 10

        # Community (15 pts)
        community = 0
        if repo.stars > 0:
            community += min(10, math.log10(repo.stars + 1) * 3)
        if repo.forks > 0:
            community += min(5, math.log10(repo.forks + 1) * 2)
        scores["community"] = community

        # Completeness (15 pts)
        completeness = 0
        if repo.has_readme:
            completeness += 7
        if repo.description and len(repo.description) > 10:
            completeness += 4
        if len(repo.topics) >= 2:
            completeness += 4
        scores["completeness"] = min(15, completeness)

        return round(sum(scores.values()), 2)

    def enumerate_org_repos(self, org: str) -> List[RepoMetadata]:
        """Enumerate all public repos for an org or user."""
        repos = []
        page = 1

        while True:
            # Try org endpoint first, fall back to user
            data = self._api_get(
                f"{self.GITHUB_API}/orgs/{org}/repos",
                params={"per_page": 100, "page": page, "type": "public"},
            )
            if data is None:
                data = self._api_get(
                    f"{self.GITHUB_API}/users/{org}/repos",
                    params={"per_page": 100, "page": page, "type": "public"},
                )
            if not data:
                break

            for item in data:
                repo = self._to_repo_metadata(item)
                if repo and repo.quality_score >= self.min_quality_score:
                    repos.append(repo)

            if len(data) < 100:
                break
            page += 1
            time.sleep(0.5)

        logger.info(f"Enumerated {len(repos)} repos from {org}")
        return repos

    def search_by_topics(self, topics: List[str], max_per_topic: int = 50) -> List[RepoMetadata]:
        """Search GitHub repos by topic tags."""
        repos = []

        for topic in topics:
            query = f"topic:{topic}"
            data = self._api_get(
                f"{self.GITHUB_API}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(100, max_per_topic),
                },
            )
            if data and "items" in data:
                for item in data["items"][:max_per_topic]:
                    repo = self._to_repo_metadata(item)
                    if repo and repo.quality_score >= self.min_quality_score:
                        repos.append(repo)

            time.sleep(self.SEARCH_DELAY)

        logger.info(f"Topic search found {len(repos)} repos across {len(topics)} topics")
        return repos

    def search_by_query(self, queries: List[str]) -> List[RepoMetadata]:
        """Run stars-based search queries."""
        repos = []

        for query in queries:
            data = self._api_get(
                f"{self.GITHUB_API}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                },
            )
            if data and "items" in data:
                for item in data["items"]:
                    repo = self._to_repo_metadata(item)
                    if repo and repo.quality_score >= self.min_quality_score:
                        repos.append(repo)

            time.sleep(self.SEARCH_DELAY)

        logger.info(f"Query search found {len(repos)} repos across {len(queries)} queries")
        return repos

    def clone_repository(self, repo: RepoMetadata) -> Optional[Path]:
        """Clone a single repository (shallow)."""
        repo_path = self.output_dir / repo.full_name.replace("/", "_")

        if repo_path.exists():
            logger.debug(f"Repository exists: {repo.full_name}")
            return repo_path

        try:
            Repo.clone_from(
                repo.clone_url,
                repo_path,
                depth=1,
                branch=repo.default_branch,
            )
            logger.info(
                f"Cloned: {repo.full_name} (score: {repo.quality_score}, "
                f"domain: {repo.category})"
            )
            self.catalog.add_repository(repo, repo_path)
            return repo_path
        except Exception as e:
            logger.error(f"Clone failed {repo.full_name}: {e}")
            return None

    def collect_all(self, max_workers: int = 4) -> Generator[Path, None, None]:
        """Orchestrate all discovery methods, deduplicate, and clone."""
        all_repos: List[RepoMetadata] = []

        # 1. Seed orgs and users
        logger.info(f"Phase 1: Enumerating {len(self.seed_orgs)} orgs + {len(self.seed_users)} users")
        for org in self.seed_orgs + self.seed_users:
            repos = self.enumerate_org_repos(org)
            all_repos.extend(repos)

        logger.info(f"Seed discovery: {len(all_repos)} repos")

        # 2. Topic-based discovery
        if self.topics:
            logger.info(f"Phase 2: Searching {len(self.topics)} topics")
            topic_repos = self.search_by_topics(self.topics)
            all_repos.extend(topic_repos)

        # 3. Stars-based search queries
        if self.search_queries:
            logger.info(f"Phase 3: Running {len(self.search_queries)} search queries")
            query_repos = self.search_by_query(self.search_queries)
            all_repos.extend(query_repos)

        # Sort by quality score descending
        all_repos.sort(key=lambda r: r.quality_score, reverse=True)

        logger.info(
            f"Total unique repos discovered: {len(all_repos)} "
            f"(deduped from {len(self._seen)} seen)"
        )

        # Log domain distribution
        domain_counts: Dict[str, int] = {}
        for r in all_repos:
            domain_counts[r.category] = domain_counts.get(r.category, 0) + 1
        logger.info(f"Domain distribution: {domain_counts}")

        # Clone in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.clone_repository, repo): repo
                for repo in all_repos
            }
            for future in as_completed(futures):
                repo_path = future.result()
                if repo_path:
                    yield repo_path
