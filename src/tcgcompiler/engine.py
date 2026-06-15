from __future__ import annotations

from collections.abc import Iterable

from .models import TemplateMatch
from .templates import TextTemplate, build_templates


class TemplateEngine:
    """Apply reusable text templates to card-effect clauses."""

    def __init__(self, templates: Iterable[TextTemplate] | None = None) -> None:
        self.templates = list(templates if templates is not None else build_templates())

    def match_first(self, text: str, *, executable_only: bool = True) -> TemplateMatch | None:
        for template in self.templates:
            match = template.try_match(text)
            if match is None:
                continue
            if executable_only and not match.executable:
                continue
            return match
        return None

    def match_all(self, text: str, *, executable_only: bool = True) -> list[TemplateMatch]:
        matches: list[TemplateMatch] = []
        for template in self.templates:
            match = template.try_match(text)
            if match is None:
                continue
            if executable_only and not match.executable:
                continue
            matches.append(match)
        return matches


def default_template_engine() -> TemplateEngine:
    return TemplateEngine()
