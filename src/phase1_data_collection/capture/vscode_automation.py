"""
phase1_data_collection/capture/vscode_automation.py

Screenshot capture using Playwright's Chromium with Monaco Editor.
Renders code files in a browser with VS Code's editor engine — no VS Code
installation or Xvfb required.
"""
import asyncio
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

# Language to Monaco language ID mapping
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".nim": "plaintext",
    ".rb": "ruby",
    ".c": "c",
}

# VS Code Dark+ inspired theme for Monaco
DARK_PLUS_THEME = {
    "base": "vs-dark",
    "inherit": True,
    "rules": [],
    "colors": {
        "editor.background": "#1e1e1e",
        "editor.foreground": "#d4d4d4",
        "editorLineNumber.foreground": "#858585",
        "editorLineNumber.activeForeground": "#c6c6c6",
    }
}

THEME_CONFIGS = {
    "Default Dark+": {"base": "vs-dark", "bg": "#1e1e1e", "fg": "#d4d4d4"},
    "Default Light+": {"base": "vs", "bg": "#ffffff", "fg": "#000000"},
    "Monokai": {"base": "vs-dark", "bg": "#272822", "fg": "#f8f8f2"},
    "Solarized Dark": {"base": "vs-dark", "bg": "#002b36", "fg": "#839496"},
    "Solarized Light": {"base": "vs", "bg": "#fdf6e3", "fg": "#657b83"},
    "GitHub Dark": {"base": "vs-dark", "bg": "#0d1117", "fg": "#c9d1d9"},
    "Dracula": {"base": "vs-dark", "bg": "#282a36", "fg": "#f8f8f2"},
    "One Dark Pro": {"base": "vs-dark", "bg": "#282c34", "fg": "#abb2bf"},
}


@dataclass
class CaptureConfig:
    """Screenshot capture configuration."""
    viewport_width: int = 2560
    viewport_height: int = 1440
    device_scale_factor: float = 2.0
    font_size: int = 14
    line_height: int = 20
    theme: str = "Default Dark+"
    scroll_step: int = 1080
    render_delay_ms: int = 100


@dataclass
class CaptureResult:
    """Result from capturing a single file."""
    file_path: Path
    screenshots: List[Path] = field(default_factory=list)
    source_code: str = ""
    file_hash: str = ""
    metadata: Dict = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None


def _build_monaco_html(config: CaptureConfig) -> str:
    """Build an HTML page with Monaco Editor embedded."""
    theme_cfg = THEME_CONFIGS.get(config.theme, THEME_CONFIGS["Default Dark+"])
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; background: {theme_cfg["bg"]}; }}
  #editor {{ width: 100%; height: 100%; }}
</style>
</head>
<body>
<div id="editor"></div>
<script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/loader.js"></script>
<script>
  require.config({{ paths: {{ vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' }} }});
  window._editorReady = false;
  require(['vs/editor/editor.main'], function() {{
    window._editor = monaco.editor.create(document.getElementById('editor'), {{
      value: '',
      language: 'plaintext',
      theme: '{theme_cfg["base"]}',
      fontSize: {config.font_size},
      lineHeight: {config.line_height},
      minimap: {{ enabled: false }},
      scrollbar: {{ vertical: 'hidden', horizontal: 'hidden' }},
      renderWhitespace: 'none',
      guides: {{ indentation: false }},
      lineNumbers: 'on',
      wordWrap: 'off',
      readOnly: true,
      automaticLayout: true,
      scrollBeyondLastLine: false,
      overviewRulerLanes: 0,
      hideCursorInOverviewRuler: true,
      overviewRulerBorder: false,
      renderLineHighlight: 'none',
      contextmenu: false,
      folding: false,
    }});
    window._editorReady = true;
  }});
</script>
</body>
</html>"""


class MonacoCapture:
    """Screenshot capture using Playwright Chromium + Monaco Editor."""

    def __init__(
        self,
        config: CaptureConfig = None,
        output_dir: Path = None,
        headless: bool = True
    ):
        self.config = config or CaptureConfig()
        self.output_dir = Path(output_dir or "./captures")
        self.headless = headless
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._html_content = _build_monaco_html(self.config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Launch headless Chromium and load Monaco Editor."""
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
            device_scale_factor=self.config.device_scale_factor
        )
        self.page = await self.context.new_page()

        # Load Monaco Editor
        await self.page.set_content(self._html_content)
        # Wait for Monaco to initialize
        await self.page.wait_for_function("() => window._editorReady === true", timeout=30000)
        logger.info("Monaco Editor initialized in headless Chromium")

    async def stop(self):
        """Clean up resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _load_code(self, source_code: str, language: str):
        """Load code into the Monaco editor."""
        # Escape for JavaScript string
        escaped = json.dumps(source_code)
        await self.page.evaluate(f"""
            () => {{
                const model = monaco.editor.createModel({escaped}, '{language}');
                window._editor.setModel(model);
                window._editor.revealLine(1);
            }}
        """)
        await asyncio.sleep(self.config.render_delay_ms / 1000)

    async def _get_scroll_info(self) -> Tuple[int, int]:
        """Get current scroll position and total content height."""
        info = await self.page.evaluate("""
            () => {
                const e = window._editor;
                return {
                    scrollTop: e.getScrollTop(),
                    scrollHeight: e.getScrollHeight(),
                    layoutHeight: e.getLayoutInfo().height
                };
            }
        """)
        return info.get("scrollTop", 0), info.get("scrollHeight", 0), info.get("layoutHeight", 0)

    async def _scroll_to(self, position: int):
        """Scroll editor to pixel position."""
        await self.page.evaluate(f"() => window._editor.setScrollTop({position})")
        await asyncio.sleep(self.config.render_delay_ms / 1000)

    async def _capture_screenshot(self, output_path: Path) -> bool:
        """Capture current viewport as WebP."""
        try:
            await self.page.screenshot(path=str(output_path), type='png')
            return True
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False

    async def capture_file(self, file_path: Path) -> CaptureResult:
        """Capture all screenshots for a code file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            file_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]
        except Exception as e:
            return CaptureResult(file_path=file_path, success=False, error=str(e))

        # Detect language
        lang = LANGUAGE_MAP.get(file_path.suffix.lower(), "plaintext")

        # Load into editor
        await self._load_code(source_code, lang)

        # Get dimensions
        scroll_top, scroll_height, layout_height = await self._get_scroll_info()
        num_captures = max(1, (scroll_height + self.config.scroll_step - 1) // self.config.scroll_step)

        capture_dir = self.output_dir / file_hash[:2] / file_hash
        capture_dir.mkdir(parents=True, exist_ok=True)

        screenshots = []
        for i in range(num_captures):
            scroll_pos = i * self.config.scroll_step
            await self._scroll_to(scroll_pos)

            screenshot_path = capture_dir / f"{i:04d}.png"
            if await self._capture_screenshot(screenshot_path):
                screenshots.append(screenshot_path)

            # Check if we've scrolled past the end
            current_top, _, _ = await self._get_scroll_info()
            if current_top + layout_height >= scroll_height:
                break

        # Save source
        with open(capture_dir / "source.txt", 'w', encoding='utf-8') as f:
            f.write(source_code)

        # Save metadata
        metadata = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "language": lang,
            "line_count": source_code.count('\n') + 1,
            "char_count": len(source_code),
            "num_screenshots": len(screenshots),
            "theme": self.config.theme,
            "font_size": self.config.font_size,
            "viewport": f"{self.config.viewport_width}x{self.config.viewport_height}"
        }

        with open(capture_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Captured {len(screenshots)} screenshots: {file_path.name}")

        return CaptureResult(
            file_path=file_path, screenshots=screenshots,
            source_code=source_code, file_hash=file_hash,
            metadata=metadata, success=True
        )


# Backward-compatible alias
VSCodeAutomation = MonacoCapture
