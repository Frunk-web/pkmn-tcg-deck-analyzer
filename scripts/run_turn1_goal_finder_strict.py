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



# -----------------------------
# v0.3 deck-resolution patches
# -----------------------------
# The coverage scanner can match alternate prints by effect source card_id/name,
# but the base goal finder resolves decklists from card-level identity only. Some
# deduped compiled cards do not expose every alternate print at card-level, so a
# valid deck can become deck_size=null / scenarios=[] after all entries resolve
# to nothing. These patches extend the runtime name index with effect.source
# metadata and make Basic Energy proxy detection handle ids like sve-6 / mee-6.

_ORIG_TF_BUILD_NAME_INDEX = gf.tf.build_name_index
_ORIG_TF_RESOLVE_DECKLIST = gf.tf.resolve_decklist
_ORIG_TF_BASIC_ENERGY_PROXY = gf.tf.basic_energy_proxy_from_request

SET_ALIASES = {
    "MEG": ["me1"],
    "POR": ["me3"],
    "ASC": ["me2pt5"],
    "TWM": ["sv6"],
    "SSP": ["sv8"],
    "SFA": ["sv6pt5"],
    "MEE": ["sve", "me1"],
    "JTG": ["sv9"],
    "BLK": ["zsv10pt5", "sv10pt5", "rsv10pt5"],
}

ENERGY_ID_HINTS = {
    "sve1": "Grass", "sve3": "Water", "sve4": "Lightning", "sve5": "Psychic", "sve6": "Fighting",
    "mee1": "Grass", "mee3": "Water", "mee4": "Lightning", "mee5": "Psychic", "mee6": "Fighting",
    "sve-1": "Grass", "sve-3": "Water", "sve-4": "Lightning", "sve-5": "Psychic", "sve-6": "Fighting",
    "mee-1": "Grass", "mee-3": "Water", "mee-4": "Lightning", "mee-5": "Psychic", "mee-6": "Fighting",
}

TYPE_MAP = {
    "{G}": "Grass", "{R}": "Fire", "{W}": "Water", "{L}": "Lightning", "{P}": "Psychic",
    "{F}": "Fighting", "{D}": "Darkness", "{M}": "Metal", "{Y}": "Fairy", "{C}": "Colorless",
}


def _clean_ptcgl_name(name: str) -> str:
    out = str(name or "").strip().lstrip("\ufeff")
    for sym, word in TYPE_MAP.items():
        out = out.replace(sym, word)
    return " ".join(out.split())


def _norm_key(value: Any) -> str:
    return gf.tf.norm(value)


def _effect_sources(card: Any) -> list[dict]:
    if not isinstance(card, dict):
        return []
    out = []
    for eff in card.get("compiled_effects", []) or card.get("effects", []) or []:
        if isinstance(eff, dict):
            src = eff.get("source") or {}
            if isinstance(src, dict):
                out.append(src)
    return out


def _source_names_and_ids(card: Any) -> set[str]:
    names = set()
    if not isinstance(card, dict):
        return names
    for src in _effect_sources(card):
        for key in ("card_id", "id", "card_name", "name"):
            val = src.get(key)
            if val:
                names.add(str(val))
    return names


def patched_build_name_index(cards: Any):
    index = _ORIG_TF_BUILD_NAME_INDEX(cards)
    for c in cards:
        for name in _source_names_and_ids(c):
            key = _norm_key(name)
            if key:
                index[key].append(c)
    return index


def _parse_ptcgl_requested(requested: str) -> tuple[str, str, str] | None:
    """Return (name, set_code, number) for strings like 'Meowth ex POR 62'."""
    parts = str(requested or "").strip().lstrip("\ufeff").split()
    if len(parts) < 3:
        return None
    number = parts[-1]
    set_code = parts[-2]
    # avoid treating API ids or short names as PTCGL rows
    if "-" in requested and len(parts) == 1:
        return None
    if not number.replace("-", "").replace("/", "").isalnum():
        return None
    if not set_code.isalnum() or len(set_code) > 8:
        return None
    name = _clean_ptcgl_name(" ".join(parts[:-2]))
    if not name:
        return None
    return name, set_code.upper(), number


def _candidate_requested_keys(requested: str) -> list[str]:
    requested = str(requested or "").strip().lstrip("\ufeff")
    keys = []
    if requested:
        keys.append(requested)
    parsed = _parse_ptcgl_requested(requested)
    if parsed:
        name, set_code, number = parsed
        keys.append(name)
        # exact alias ids when known, e.g. MEG 77 -> me1-77
        for alias in SET_ALIASES.get(set_code, [set_code.lower()]):
            keys.append(f"{alias}-{number}")
    return keys


def patched_basic_energy_proxy_from_request(requested: str):
    raw = str(requested or "").strip().lstrip("\ufeff")
    raw_lower = raw.lower()
    compact = "".join(ch for ch in raw_lower if ch.isalnum())
    typ = ENERGY_ID_HINTS.get(raw_lower) or ENERGY_ID_HINTS.get(compact)
    if typ:
        name = f"Basic {typ} Energy"
        proxy_id = f"proxy-energy-{compact or typ.lower()}"
        return {
            "card_id": proxy_id,
            "representative_card_id": proxy_id,
            "identity": {
                "card_id": proxy_id,
                "name": name,
                "canonical_name": name,
                "supertype": "Energy",
                "subtypes": ["Basic"],
                "types": [typ],
            },
            "gameplay": {},
            "compiled_effects": [],
            "parser_status": "proxy_basic_energy",
            "source": {"proxy_for_decklist_entry": raw},
            "same_effect_printings": [{"card_id": raw, "id": raw, "name": name}],
        }
    parsed = _parse_ptcgl_requested(raw)
    if parsed:
        name, set_code, number = parsed
        # Convert PTCGL-style Basic {F} Energy MEE 6 into a normal energy proxy.
        if "basic" in name.lower() and "energy" in name.lower():
            return patched_basic_energy_proxy_from_request(name)
    return _ORIG_TF_BASIC_ENERGY_PROXY(requested)


def patched_resolve_decklist(decklist: Any, cards: Any):
    index = patched_build_name_index(cards)
    all_keys = list(index.keys())
    deck = []
    unresolved = []
    for count, requested in decklist:
        candidates = []
        tried_keys = []
        for candidate in _candidate_requested_keys(requested):
            key = _norm_key(candidate)
            if not key or key in tried_keys:
                continue
            tried_keys.append(key)
            candidates = index.get(key) or []
            if candidates:
                break
        if not candidates:
            # conservative substring fallback, same spirit as base resolver
            for key in tried_keys:
                hits = [k for k in all_keys if key and key in k]
                unique = []
                seen = set()
                for h in hits:
                    for c in index[h]:
                        cid = gf.tf.card_id(c)
                        if cid not in seen:
                            unique.append(c)
                            seen.add(cid)
                if len(unique) == 1:
                    candidates = unique
                    break
        if not candidates:
            proxy = patched_basic_energy_proxy_from_request(str(requested))
            if proxy is not None:
                for _ in range(int(count)):
                    deck.append(proxy)
                continue
            unresolved.append({"requested_name": requested, "count": count, "tried_keys": tried_keys})
            continue
        chosen = candidates[0]
        for _ in range(int(count)):
            deck.append(chosen)
    return deck, unresolved


def install_deck_resolution_patches() -> None:
    gf.tf.build_name_index = patched_build_name_index
    gf.tf.resolve_decklist = patched_resolve_decklist
    gf.tf.basic_energy_proxy_from_request = patched_basic_energy_proxy_from_request


_orig_score_candidate_for_missing_targets = gf.score_candidate_for_missing_targets
_orig_execute_action = gf.execute_action

VERSION = "turn1_strict_runtime_wrapper_v0_3"


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
    install_deck_resolution_patches()
    gf.score_candidate_for_missing_targets = strict_score_candidate_for_missing_targets
    gf.execute_action = strict_execute_action
    setattr(gf, "STRICT_TURN1_RUNTIME_SEMANTICS_VERSION", VERSION)


install()



# ---------------------------------------------------------------------
# TURN1_UNRESOLVED_TO_INERT_PROXY_V08
# ---------------------------------------------------------------------
# For Streamlit runs, unresolved non-goal deck cards should not kill the
# whole Turn-1 simulation. They become inert placeholder cards:
# - still occupy deck slots
# - can be drawn/prized
# - have no executable effects
# - conservative instead of inflated

_ORIG_RESOLVE_DECKLIST_AFTER_STRICT_PATCHES_V08 = gf.tf.resolve_decklist

def _turn1_v08_clean_display_name(requested):
    raw = str(requested or "").strip().lstrip(chr(65279))

    try:
        parsed = _parse_ptcgl_requested(raw)
        if parsed:
            name, _set_code, _number = parsed
            return name
    except Exception:
        pass

    try:
        cleaned = _clean_ptcgl_name(raw)
        if cleaned:
            return cleaned
    except Exception:
        pass

    # Generic fallback: remove final set/number tokens if present.
    parts = raw.split()
    if len(parts) >= 3 and parts[-1].isdigit():
        return " ".join(parts[:-2]) or raw

    return raw or "Unknown unresolved card"


def _turn1_v08_inert_unresolved_proxy(requested):
    raw = str(requested or "").strip().lstrip(chr(65279))
    clean_name = _turn1_v08_clean_display_name(raw)

    compact = "".join(ch.lower() for ch in raw if ch.isalnum())[:80] or "unknown"
    proxy_id = f"proxy-unresolved-{compact}"

    return {
        "card_id": proxy_id,
        "representative_card_id": proxy_id,
        "identity": {
            "card_id": proxy_id,
            "name": clean_name,
            "canonical_name": clean_name,
            "supertype": "Unresolved",
            "subtypes": [],
            "types": [],
        },
        "gameplay": {},
        "compiled_effects": [],
        "effects": [],
        "parser_status": "proxy_unresolved_inert",
        "source": {
            "proxy_for_decklist_entry": raw,
            "proxy_reason": "unresolved_deck_card_treated_as_inert_placeholder",
        },
        "same_effect_printings": [
            {
                "card_id": raw,
                "id": raw,
                "name": clean_name,
            }
        ],
    }


def _turn1_v08_resolve_decklist_allow_inert_unresolved(decklist, cards):
    deck, unresolved = _ORIG_RESOLVE_DECKLIST_AFTER_STRICT_PATCHES_V08(decklist, cards)

    if not unresolved:
        return deck, unresolved

    converted = []

    for item in unresolved:
        requested = item.get("requested_name") or item.get("name") or "Unknown unresolved card"
        count = int(item.get("count") or 0)

        if count <= 0:
            continue

        proxy = _turn1_v08_inert_unresolved_proxy(requested)

        for _ in range(count):
            deck.append(proxy)

        converted.append(
            {
                "requested_name": requested,
                "count": count,
                "policy": "converted_to_inert_placeholder",
                "note": (
                    "This card was not found in the compiled semantics file, so it was "
                    "kept as a blank deck card instead of failing the Turn-1 simulation."
                ),
            }
        )

    try:
        setattr(gf, "TURN1_V08_INERT_UNRESOLVED_CARDS", converted)
    except Exception:
        pass

    return deck, []


gf.tf.resolve_decklist = _turn1_v08_resolve_decklist_allow_inert_unresolved
setattr(gf, "TURN1_UNRESOLVED_TO_INERT_PROXY_VERSION", "v0.8")

if __name__ == "__main__":
    gf.main()
