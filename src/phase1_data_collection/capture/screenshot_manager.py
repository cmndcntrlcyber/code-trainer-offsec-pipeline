"""
phase1_data_collection/capture/screenshot_manager.py

High-level screenshot management coordinating capture and catalog updates.
"""
import asyncio
import logging
from pathlib import Path
from typing import List

from ..scrapers.sqlite_catalog import SQLiteCatalog
from .vscode_automation import CaptureConfig, CaptureResult
from .parallel_capture import ParallelCapture
from .theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """Coordinates screenshot capture with catalog tracking."""

    def __init__(
        self,
        catalog: SQLiteCatalog,
        output_dir: Path,
        config: CaptureConfig = None,
        num_workers: int = 8,
        rotate_themes: bool = False
    ):
        self.catalog = catalog
        self.output_dir = Path(output_dir)
        self.config = config or CaptureConfig()
        self.num_workers = num_workers
        self.theme_manager = ThemeManager() if rotate_themes else None

    def capture_files(
        self,
        file_paths: List[Path],
        repo_name: str
    ) -> List[CaptureResult]:
        """Capture screenshots for a list of files and update catalog."""
        if self.theme_manager:
            theme = self.theme_manager.rotate_and_apply()
            self.config.theme = theme

        capture = ParallelCapture(
            num_workers=self.num_workers,
            config=self.config,
            output_dir=self.output_dir
        )

        results = asyncio.run(capture.capture_all(file_paths))

        for result in results:
            if result.success:
                self.catalog.add_capture(
                    repo_name=repo_name,
                    file_path=result.file_path,
                    file_hash=result.file_hash,
                    language=result.metadata.get("language", ""),
                    line_count=result.metadata.get("line_count", 0),
                    screenshot_count=len(result.screenshots),
                    metadata=result.metadata
                )

        successful = sum(1 for r in results if r.success)
        logger.info(f"Captured {successful}/{len(results)} files for {repo_name}")

        return results
