from __future__ import annotations
"""
Strict Turn-1 goal-finder wrapper v0.2-runtime-wrapper-fix.

Why this exists:
- Earlier generic-runtime wrapper used runpy.run_path(), which executes a fresh
  __main__ copy of run_turn1_goal_finder.py. Monkey patches applied to the
  imported module did not necessarily affect the executed module.
- This version imports run_turn1_goal_finder as a module, patches that module,
  and calls gf.main().
- Run Errand is treated as an Ability, not an attack: it draws 2 random cards,
  does not end the turn, and can be followed by other legal actions.
- The wrapper filters only candidate actions; it does not rewrite card effects.
"""

import os
import sys
from typing import Any, Iterable, List

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_turn1_goal_finder as gf  # type: ignore

_orig_score_candidate_for_missing_targets = gf.score_candidate_for_missing_targets
_orig_execute_action = gf.execute_action

VERSION = "turn1_strict_runtime_wrapper_v0_2"


def _norm_text(x: Any) -> str:
    return str(x or "").strip().lower()


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _walk_strings(x)


def _contains_semantic_ir_marker(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj == "semantic_ir_marker"
    if isinstance(obj, dict):
        if obj.get("op") == "semantic_ir_marker":
            return True
        return any(_contains_semantic_ir_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_semantic_ir_marker(v) for v in obj)
    return False


def _card_strings(card: Any) -> List[str]:
    out: List[str] = []
    if isinstance(card, dict):
        for key in ("name", "card_name", "card_id", "id"):
            if card.get(key):
                out.append(str(card.get(key)))
        for s in _walk_strings(card):
            if s and len(s) < 500:
                out.append(s)
    elif card is not None:
        out.append(str(card))
    return out


def _short_dict_strings(obj: Any) -> List[str]:
    """Strings describing the candidate action/effect, not the whole state."""
    out: List[str] = []
    if not isinstance(obj, dict):
        if obj is not None:
            out.append(str(obj))
        return out

    scalar_keys = (
        "_virtual_action", "name", "label", "action", "action_name",
        "card_name", "source_name", "source_card_name", "effect_name",
        "ability_name", "attack_name", "target_name", "op",
    )
    for key in scalar_keys:
        val = obj.get(key)
        if isinstance(val, (str, int, float, bool)):
            out.append(str(val))

    nested_keys = ("card", "source", "source_card", "effect", "compiled_effect", "attack", "ability")
    nested_scalar_keys = ("name", "label", "title", "text", "effect_text", "ability_text", "attack_text", "op")
    for key in nested_keys:
        val = obj.get(key)
        if isinstance(val, dict):
            for nk in nested_scalar_keys:
                nv = val.get(nk)
                if isinstance(nv, (str, int, float, bool)):
                    out.append(str(nv))
        elif isinstance(val, (str, int, float, bool)):
            out.append(str(val))
    return out


def _action_strings(action: Any) -> List[str]:
    out: List[str] = []
    try:
        out.append(gf.action_label(action))
    except Exception:
        out.append(str(action))

    if isinstance(action, dict):
        out.extend(_short_dict_strings(action))
    else:
        out.extend(_card_strings(action))
    return [s for s in out if s and len(str(s)) < 500]


def _candidate_payloads(action: Any) -> List[Any]:
    if not isinstance(action, dict):
        return [action]
    out: List[Any] = []
    for key in ("op", "ops", "steps", "compiled_steps", "effect", "compiled_effect", "card", "source"):
        if key in action:
            out.append(action.get(key))
    return out or [action]


def _effect_runtime(action: Any) -> dict:
    """Best-effort turn1_runtime lookup on a candidate action/effect/card."""
    if isinstance(action, dict):
        if isinstance(action.get("turn1_runtime"), dict):
            return action["turn1_runtime"]
        for key in ("effect", "compiled_effect"):
            val = action.get(key)
            if isinstance(val, dict) and isinstance(val.get("turn1_runtime"), dict):
                return val["turn1_runtime"]
        # If the action is a whole card, inspect its effects. Only return a
        # trusted runtime if all semantic effects are trusted; otherwise empty.
        effects = action.get("compiled_effects") if isinstance(action.get("compiled_effects"), list) else None
        if effects:
            runtimes = [e.get("turn1_runtime") for e in effects if isinstance(e, dict) and isinstance(e.get("turn1_runtime"), dict)]
            trusted = [rt for rt in runtimes if _runtime_is_trusted(rt)]
            return {"card_runtime_count": len(runtimes), "card_trusted_runtime_count": len(trusted)} if runtimes else {}
    return {}


def _runtime_is_trusted(rt: Any) -> bool:
    if not isinstance(rt, dict):
        return False
    if rt.get("exact_runtime_support") is True and rt.get("runtime_support") != "not_executable_without_more_structure":
        return True
    # Structured-from-compiled-text is safer than broad semantic fallback because
    # it comes from already compiled ops such as search_deck/draw_cards.
    if rt.get("runtime_support") == "structured_from_compiled_text":
        return True
    return False


def _candidate_effect_has_semantic_ir_marker(action: Any) -> bool:
    # For a candidate whole card, scan compiled effects. For virtual actions,
    # scan the nested candidate effect/source only.
    if isinstance(action, str):
        return action == "semantic_ir_marker"
    if not isinstance(action, dict):
        return False
    return any(_contains_semantic_ir_marker(p) for p in _candidate_payloads(action))


def _candidate_semantic_runtime_trusted(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    # Virtual actions implemented directly by the target finder are trusted here;
    # they are not executing broad semantic fallback blindly.
    if action.get("_virtual_action") in {"Run Errand", "Teal Dance", "BenchAbility", "AbilityRequirementSearch"}:
        return True
    rt = _effect_runtime(action)
    if _runtime_is_trusted(rt):
        return True
    # Whole-card case: only trust if every semantic marker on the card has some
    # trusted turn1_runtime. If not, block the candidate rather than guessing.
    effects = action.get("compiled_effects") if isinstance(action.get("compiled_effects"), list) else None
    if effects:
        semantic_effects = [e for e in effects if isinstance(e, dict) and _contains_semantic_ir_marker(e.get("steps", []))]
        if not semantic_effects:
            return True
        return all(_runtime_is_trusted(e.get("turn1_runtime")) for e in semantic_effects if isinstance(e, dict))
    return False


def _active_has_run_errand(st: Any) -> bool:
    active = getattr(st, "active", None)
    if active is None:
        return False
    return any("run errand" in _norm_text(s) for s in _card_strings(active))


def _is_run_errand_action(action: Any) -> bool:
    if isinstance(action, dict) and action.get("_virtual_action") == "Run Errand":
        return True
    return any(_norm_text(s) == "run errand" or "run errand" in _norm_text(s) for s in _action_strings(action))


def _blocked_reason(st: Any, action: Any, going: str) -> str | None:
    strings = [_norm_text(s) for s in _action_strings(action)]
    blob = "\n".join(strings)

    if "unfair stamp" in blob:
        return "blocked_unfair_stamp_condition_not_satisfied"
    if "flip the script" in blob or "fezandipiti ex" in blob:
        return "blocked_flip_the_script_requires_ko_last_turn"

    # Run Errand is an Ability: do not make it terminal, do not require going second.
    # The base target finder already requires Mega Kangaskhan ex to be Active.
    # Keep the check here so a generic candidate cannot sneak through.
    if _is_run_errand_action(action):
        if not _active_has_run_errand(st):
            return "blocked_run_errand_requires_active_source"
        return None

    # Semantic fallback is not executable unless the candidate has trusted
    # structured runtime support. This prevents broad semantic-only cards from
    # silently inflating probabilities.
    if _candidate_effect_has_semantic_ir_marker(action) and not _candidate_semantic_runtime_trusted(action):
        return "blocked_semantic_ir_without_trusted_runtime"

    return None


def strict_score_candidate_for_missing_targets(st: Any, missing: Any, going: str, enable_chain_search: bool):
    scored = _orig_score_candidate_for_missing_targets(st, missing, going, enable_chain_search)
    kept = []
    for score, action, target_norm in scored:
        reason = _blocked_reason(st, action, going)
        if reason is None:
            kept.append((score, action, target_norm))
        else:
            try:
                st.log.append({
                    "event": "strict_runtime_block",
                    "version": VERSION,
                    "reason": reason,
                    "action": gf.action_label(action),
                    "going": going,
                    "active": gf.tf.card_name(getattr(st, "active", None)) if getattr(st, "active", None) is not None else None,
                })
            except Exception:
                pass
    return kept


def strict_execute_action(st: Any, action: Any, target_norm: str, rng: Any, going: str, enable_chain_search: bool) -> None:
    # Do not terminalize Run Errand. It is an Ability and can be followed by
    # later legal actions.
    _orig_execute_action(st, action, target_norm, rng, going, enable_chain_search)


def install() -> None:
    gf.score_candidate_for_missing_targets = strict_score_candidate_for_missing_targets
    gf.execute_action = strict_execute_action
    setattr(gf, "STRICT_TURN1_RUNTIME_SEMANTICS_VERSION", VERSION)


install()

if __name__ == "__main__":
    gf.main()
