from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


Step = dict[str, Any]
Builder = Callable[[re.Match[str], str], list[Step]]


@dataclass(frozen=True)
class TemplateMatch:
    """A parsed template result.

    executable=True means the emitted steps are intended for strict compiler
    completeness. Non-executable/recognition-only templates belong in reporting,
    not in compile_cards_auto.py complete status.
    """

    family: str
    template_id: str
    source_text: str
    steps: list[Step]
    confidence: float = 0.9
    executable: bool = True
    notes: list[str] = field(default_factory=list)
