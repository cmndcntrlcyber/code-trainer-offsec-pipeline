"""
phase1_data_collection/capture/parallel_capture.py

Parallel screenshot capture with multiple headless Chromium instances.
"""
import asyncio
import logging
from pathlib import Path
from typing import List

from .vscode_automation import MonacoCapture, CaptureConfig, CaptureResult

logger = logging.getLogger(__name__)


class ParallelCapture:
    """Parallel screenshot capture with multiple browser instances."""

    def __init__(
        self,
        num_workers: int = 8,
        config: CaptureConfig = None,
        output_dir: Path = None
    ):
        self.num_workers = num_workers
        self.config = config or CaptureConfig()
        self.output_dir = Path(output_dir or "./captures")

    async def capture_batch(self, file_paths: List[Path], worker_id: int) -> List[CaptureResult]:
        """Process files with a single browser instance."""
        capture = MonacoCapture(
            config=self.config,
            output_dir=self.output_dir,
            headless=True
        )

        results = []
        try:
            await capture.start()
            for file_path in file_paths:
                try:
                    result = await capture.capture_file(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed on {file_path.name}: {e}")
                    results.append(CaptureResult(file_path=file_path, success=False, error=str(e)))
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to start: {e}")
            for fp in file_paths:
                results.append(CaptureResult(file_path=fp, success=False, error=str(e)))
        finally:
            await capture.stop()

        return results

    async def capture_all(self, file_paths: List[Path]) -> List[CaptureResult]:
        """Capture all files using parallel workers."""
        if not file_paths:
            return []

        # Don't use more workers than files
        num_workers = min(self.num_workers, len(file_paths))

        batch_size = (len(file_paths) + num_workers - 1) // num_workers
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

        tasks = [self.capture_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)

        all_results = [r for batch in batch_results for r in batch]
        successful = sum(1 for r in all_results if r.success)
        logger.info(f"Captured {successful}/{len(all_results)} files successfully")

        return all_results
