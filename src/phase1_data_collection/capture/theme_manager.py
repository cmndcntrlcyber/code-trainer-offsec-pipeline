"""
phase1_data_collection/capture/theme_manager.py

Theme rotation for capture diversity in training data.
"""
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class ThemeManager:
    """Rotates VS Code themes for training data diversity."""

    DEFAULT_THEMES = [
        "Default Dark+",
        "Default Light+",
        "Monokai",
        "Solarized Dark",
        "Solarized Light",
        "One Dark Pro",
        "Dracula",
        "GitHub Dark",
    ]

    def __init__(self, themes: List[str] = None):
        self.themes = themes or self.DEFAULT_THEMES
        self._index = 0

    def next_theme(self) -> str:
        """Get the next theme in rotation."""
        theme = self.themes[self._index]
        self._index = (self._index + 1) % len(self.themes)
        return theme

    def apply_theme(self, theme: str):
        """Apply a VS Code theme by updating settings."""
        settings_path = Path.home() / ".config/Code/User/settings.json"

        try:
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            else:
                settings = {}

            settings["workbench.colorTheme"] = theme

            settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)

            logger.info(f"Applied theme: {theme}")
        except Exception as e:
            logger.error(f"Failed to apply theme {theme}: {e}")

    def rotate_and_apply(self) -> str:
        """Rotate to next theme and apply it."""
        theme = self.next_theme()
        self.apply_theme(theme)
        return theme
