"""Template-driven compiler helpers for Pokémon TCG card text.

This package is intentionally independent from scripts/compile_cards_auto.py for now.
The first milestone is to make effect parsing modular and testable before routing
production compilation through it.
"""

from .engine import TemplateEngine, default_template_engine
from .models import TemplateMatch

__all__ = ["TemplateEngine", "TemplateMatch", "default_template_engine"]
