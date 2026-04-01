"""
phase1_data_collection/scrapers/quality_scorer.py

Score repository quality 0-100 based on multiple factors.

Components (20 points each):
- Stars (log scale)
- Activity (recent updates)
- Documentation (readme, description, topics)
- Code quality (issues ratio, license)
- Community (forks, topics)
"""
import math
from datetime import datetime


class QualityScorer:
    """Score repository quality 0-100 based on multiple factors."""

    CATEGORIES = {
        "security": ["security", "auth", "crypto", "encryption", "vulnerability", "pentest"],
        "ai_ml": ["machine-learning", "deep-learning", "neural", "ai", "nlp", "transformer", "llm"],
        "web": ["web", "frontend", "backend", "api", "rest", "graphql", "react", "vue", "django"],
        "automation": ["automation", "ci-cd", "devops", "kubernetes", "docker", "terraform"],
        "data": ["data", "analytics", "etl", "pipeline", "database", "sql", "pandas"],
        "tool": ["cli", "tool", "utility", "library", "framework", "sdk"]
    }

    @classmethod
    def score_repository(cls, repo) -> float:
        """Calculate quality score 0-100."""
        scores = {}

        # Stars (log scale, max 20 at 10K+ stars)
        if repo.stars > 0:
            scores["stars"] = min(20, math.log10(repo.stars + 1) * 5)
        else:
            scores["stars"] = 0

        # Activity (recent updates)
        try:
            updated = datetime.fromisoformat(repo.updated_at.replace("Z", "+00:00"))
            days_since = (datetime.now(updated.tzinfo) - updated).days
            if days_since < 30:
                scores["activity"] = 20
            elif days_since < 90:
                scores["activity"] = 15
            elif days_since < 180:
                scores["activity"] = 10
            elif days_since < 365:
                scores["activity"] = 5
            else:
                scores["activity"] = 0
        except (ValueError, TypeError):
            scores["activity"] = 10

        # Documentation
        doc_score = 0
        if repo.has_readme:
            doc_score += 10
        if repo.description and len(repo.description) > 20:
            doc_score += 5
        if len(repo.topics) >= 3:
            doc_score += 5
        scores["documentation"] = doc_score

        # Code quality proxies
        quality_score = 10
        if repo.license:
            quality_score += 5
        if repo.open_issues < repo.stars * 0.1:
            quality_score += 5
        scores["code_quality"] = min(20, quality_score)

        # Community
        community_score = 0
        if repo.forks > 0:
            community_score += min(10, math.log10(repo.forks + 1) * 3)
        if len(repo.topics) > 0:
            community_score += min(10, len(repo.topics) * 2)
        scores["community"] = community_score

        return round(sum(scores.values()), 2)

    @classmethod
    def classify_category(cls, repo) -> str:
        """Classify repository into category."""
        text = f"{repo.description} {' '.join(repo.topics)}".lower()

        category_scores = {}
        for category, keywords in cls.CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=category_scores.get)
        return "general"
