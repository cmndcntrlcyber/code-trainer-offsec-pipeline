"""
phase1_data_collection/scrapers/file_filter.py

Filter code files for screenshot capture.
"""
import os
from pathlib import Path
from typing import Generator


class FileFilter:
    """Filter code files for screenshot capture."""

    EXTENSIONS = {
        "Python": [".py"],
        "JavaScript": [".js", ".jsx"],
        "TypeScript": [".ts", ".tsx"],
        "Java": [".java"],
        "C++": [".cpp", ".hpp", ".cc", ".h"],
        "Go": [".go"],
        "Rust": [".rs"],
        "C#": [".cs"]
    }

    SKIP_DIRS = {
        "node_modules", "__pycache__", ".git", ".svn",
        "vendor", "dist", "build", ".venv", "venv",
        "target", "bin", "obj", ".idea", ".vscode",
        "test", "tests", "__tests__", "spec"
    }

    MIN_FILE_SIZE = 200
    MAX_FILE_SIZE = 50_000
    MIN_LINES = 20
    MAX_LINES = 500

    @classmethod
    def get_code_files(cls, repo_path: Path) -> Generator[Path, None, None]:
        """Find valid code files."""
        all_extensions = set()
        for exts in cls.EXTENSIONS.values():
            all_extensions.update(exts)

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in cls.SKIP_DIRS]

            for filename in files:
                file_path = Path(root) / filename

                if file_path.suffix.lower() not in all_extensions:
                    continue

                try:
                    size = file_path.stat().st_size
                    if not (cls.MIN_FILE_SIZE <= size <= cls.MAX_FILE_SIZE):
                        continue

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        line_count = content.count('\n') + 1

                    if cls.MIN_LINES <= line_count <= cls.MAX_LINES:
                        yield file_path
                except (OSError, PermissionError):
                    continue


class OffSecFileFilter(FileFilter):
    """File filter with relaxed constraints and extra extensions for offensive security code."""

    EXTENSIONS = {
        **FileFilter.EXTENSIONS,
        "Python": [".py"],
        "C": [".c"],
        "PowerShell": [".ps1", ".psm1", ".psd1"],
        "Shell": [".sh", ".bash", ".zsh"],
        "Nim": [".nim"],
        "Ruby": [".rb"],
    }

    MIN_FILE_SIZE = 100
    MAX_FILE_SIZE = 100_000
    MIN_LINES = 10
    MAX_LINES = 1000
