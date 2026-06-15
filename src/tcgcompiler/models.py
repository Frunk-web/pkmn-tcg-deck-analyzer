from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


Step = dict[str, Any]
Predicate = Callable[[str], bool]
Builder = Callable[[re.Match[str], str], list[Step]]  # type: ignore[name-defined]


@dataclass(frozen=True)
class TemplateMatch:
    """A parsed template result.

    matched=True means the template understood the clause enough to emit steps.
    executable=True means the emitted steps are intended for simulator execution.
    executable=False is allowed for future analyzer-only use, but the strict compiler
    should not mark a card complete solely from non-executable matches.
    """

    family: str
    template_id: str
    source_text: str
    steps: list[Step]
    confidence: float = 0.9
    executable: bool = True
    notes: list[str] = field(default_factory=list)
