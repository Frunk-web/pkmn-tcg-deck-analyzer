"""Template-driven compiler helpers for Pokémon TCG card text.

These helpers are intentionally modular so scripts/compile_cards_auto.py can
route long-tail clauses through reusable templates instead of growing one-off
regex blocks forever.
"""

from .engine import TemplateEngine, default_template_engine
from .models import TemplateMatch

__all__ = ["TemplateEngine", "TemplateMatch", "default_template_engine"]
