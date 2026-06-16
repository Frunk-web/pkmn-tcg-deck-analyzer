from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VERSION = "runtime_coverage_scanner_v0_6"

TURN1_RELEVANT_FAMILIES = {
    "random_draw",
    "shuffle_then_draw",
    "deck_search_to_hand",
    "deck_search_to_bench",
    "topdeck_setup",
    "top_n_choose",
    "conditional_attach_then_draw",
    "bench_from_hand_trigger",
    "activated_ability",
    "supporter_rule",
    "costed_draw_or_search",
    "move_to_hand",
    "discard_recovery_to_hand",
}

IGNORABLE_FAMILIES = {
    "damage_only_attack",
    "attack_effect",
    "knockout_prize_rule",
    "continuous_battle_modifier",
    "status_condition",
    "attack_damage_scaling",
    "energy_move",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def load_card_name_map(path: Optional[Path]) -> Dict[str, str]:
    """Load card_id -> card name from all_cards.csv when available.

    The compiled corpus is deduplicated across alternate prints, so a deck may
    contain me1-104 while the compiled effect source is me1-182. The names are
    identical and the text/effect group is what matters for Turn-1 coverage.
    This map lets deck coverage match by exact card id first and by resolved
    card name second.
    """
    if not path or not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        id_col = "card_id" if "card_id" in cols else ("id" if "id" in cols else "")
        name_col = "name" if "name" in cols else ("card_name" if "card_name" in cols else "")
        if not id_col or not name_col:
            return {}
        for row in reader:
            cid = str(row.get(id_col) or "").strip()
            nm = str(row.get(name_col) or "").strip()
            if cid and nm:
                out[cid] = nm
    return out


def normalize(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def get_cards(compiled: Any) -> List[Dict[str, Any]]:
    if isinstance(compiled, list):
        return [c for c in compiled if isinstance(c, dict)]
    if isinstance(compiled, dict):
        for key in ("cards", "compiled_cards", "data", "items"):
            if isinstance(compiled.get(key), list):
                return [c for c in compiled[key] if isinstance(c, dict)]
        # fallback: dict keyed by card id
        vals = list(compiled.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
    raise ValueError("Could not find a list of cards in compiled JSON")


def effect_text(effect: Dict[str, Any]) -> str:
    source = effect.get("source") or {}
    pieces = [
        source.get("name"),
        source.get("text"),
        effect.get("text"),
        effect.get("source_text"),
    ]
    for step in effect.get("steps") or []:
        if isinstance(step, dict):
            pieces.append(step.get("source_text"))
    return "\n".join(str(p) for p in pieces if p)


def effect_name(effect: Dict[str, Any]) -> str:
    src = effect.get("source") or {}
    return str(src.get("name") or effect.get("name") or "")


def source_section(effect: Dict[str, Any]) -> str:
    return str((effect.get("source") or {}).get("section") or "")


def source_card_id(effect: Dict[str, Any]) -> str:
    return str((effect.get("source") or {}).get("card_id") or "")


def ops(effect: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for step in effect.get("steps") or []:
        if isinstance(step, dict):
            op = step.get("op")
            if op:
                out.append(str(op))
    return out


def has_op(effect: Dict[str, Any], *wanted: str) -> bool:
    have = set(ops(effect))
    return any(w in have for w in wanted)


def any_step(effect: Dict[str, Any], pred) -> bool:
    for step in effect.get("steps") or []:
        if isinstance(step, dict) and pred(step):
            return True
    return False


def destination_strings(effect: Dict[str, Any]) -> str:
    vals: List[str] = []
    for step in effect.get("steps") or []:
        if isinstance(step, dict):
            for key in ("destination", "target", "source"):
                if key in step:
                    vals.append(str(step.get(key)))
    return normalize(" ".join(vals))


def runtime_obj(effect: Dict[str, Any]) -> Dict[str, Any]:
    tr = effect.get("turn1_runtime")
    if isinstance(tr, dict):
        return tr
    rt = effect.get("runtime")
    if isinstance(rt, dict):
        return rt
    return {}


def runtime_support(effect: Dict[str, Any]) -> str:
    rt = runtime_obj(effect)
    for k in ("runtime_support", "support", "status"):
        if rt.get(k):
            return str(rt.get(k))
    # Some earlier patches store fields on the effect itself.
    for k in ("runtime_support", "turn1_runtime_support"):
        if effect.get(k):
            return str(effect.get(k))
    return ""


def runtime_patches(effect: Dict[str, Any]) -> List[str]:
    rt = runtime_obj(effect)
    patches = rt.get("patches") or rt.get("patch_ids") or rt.get("families") or []
    if isinstance(patches, str):
        return [patches]
    if isinstance(patches, list):
        return [str(x) for x in patches]
    return []


def has_semantic_marker(effect: Dict[str, Any]) -> bool:
    return has_op(effect, "semantic_ir_marker")


def classify_families(card: Dict[str, Any], effect: Dict[str, Any]) -> List[str]:
    text = normalize(effect_text(effect))
    kind = normalize(effect.get("kind"))
    section = normalize(source_section(effect))
    op_set = set(ops(effect))
    dest = destination_strings(effect)
    families: set[str] = set()

    # Core access families.
    if "draw_cards" in op_set:
        families.add("random_draw")
        if "shuffle" in text and "draw" in text:
            families.add("shuffle_then_draw")
    if "search_deck" in op_set:
        if "bench" in dest or "onto your bench" in text or "put it onto your bench" in text or "put them onto your bench" in text:
            families.add("deck_search_to_bench")
        elif "top" in dest or "on top" in text or "top of" in text:
            families.add("topdeck_setup")
        else:
            families.add("deck_search_to_hand")
    # Choice is not always card-access. Choosing a source Energy / choosing a target
    # should not be lumped into top-N card selection. Keep top_n_choose for effects
    # that actually inspect a hidden/ordered set or choose cards to move into an
    # access zone such as hand/bench/topdeck.
    if "move_energy" in op_set or "move_attached_energy" in op_set:
        families.add("energy_move")
    elif (
        "look_at_cards" in op_set
        or "look_at_top_cards" in op_set
        or re.search(r"look at the (top|bottom) \d+", text)
        or (
            "choose_cards" in op_set
            and ({"move_card", "move_cards"} & op_set)
            and ("hand" in dest or "bench" in dest or "top" in dest or "discard" in dest or "into your hand" in text or "onto your bench" in text)
        )
    ):
        families.add("top_n_choose")
    if "attach_card" in op_set and "draw_cards" in op_set:
        if "if you attached" in text or "if you attach" in text or "attached energy" in text:
            families.add("conditional_attach_then_draw")
    if "move_card" in op_set and ("self.deck.top" in dest or "deck.top" in dest or "on top" in text):
        families.add("topdeck_setup")
    if "move_cards" in op_set or "move_card" in op_set or "move_zone_to_zone" in op_set:
        if "hand" in dest or "put" in text and "into your hand" in text:
            families.add("move_to_hand")

    # Ability / trigger families.
    if "ability" in kind or section == "abilities":
        if "once during your turn" in text or kind in {"ability_activated", "activated ability"}:
            families.add("activated_ability")
        if "play this pokémon from your hand onto your bench" in text or "when you play this pokémon from your hand onto your bench" in text:
            families.add("bench_from_hand_trigger")
        if "active spot" in text:
            families.add("requires_active_source")
        if "bench" in text and "active spot" not in text:
            families.add("bench_condition")
        if "can't use more than 1" in text or "cannot use more than 1" in text:
            families.add("named_once_per_turn_limit")

    # Trainer / Supporter families.
    if "supporter" in text or kind == "supporter" or "supporter" in kind:
        families.add("supporter_rule")
    if ("you can use this card only if" in text or "you may use this card only if" in text or "were knocked out during your opponent's last turn" in text or "opponent took" in text and "prize" in text):
        families.add("conditional_comeback_guard")
    if "discard" in text and ("draw" in text or "search" in text):
        families.add("costed_draw_or_search")
    if "from your discard pile" in text and "into your hand" in text:
        families.add("discard_recovery_to_hand")

    # Battle / low Turn-1-access relevance families. In this scanner, Turn-1
    # relevance means access to cards/board pieces before attacking. Attack text
    # can matter for a future attack-readiness model, but it should not pollute
    # the card-access backlog.
    if "declare_attack" in op_set:
        families.add("attack_effect")
    if "register_knockout_prize_rule" in op_set:
        families.add("knockout_prize_rule")
    if "deal_attack_damage" in op_set and not ({"draw_cards", "search_deck"} & op_set):
        families.add("damage_only_attack")
    if "modify_attack_damage" in op_set or "modify_attack_damage_per_coin_heads" in op_set:
        families.add("attack_damage_scaling")
    if "special_condition" in text or "paralyzed" in text or "poisoned" in text or "asleep" in text or "confused" in text or "burned" in text:
        families.add("status_condition")
    if "register_continuous_modifier" in op_set and not (families & TURN1_RELEVANT_FAMILIES):
        families.add("continuous_battle_modifier")

    # Runtime patches already discovered by another pass.
    for p in runtime_patches(effect):
        if p:
            families.add(f"runtime_patch:{p}")

    return sorted(families) or ["unclassified"]


def is_turn1_relevant(families: Sequence[str], text: str, op_list: Sequence[str]) -> bool:
    fam_base = {f for f in families if not f.startswith("runtime_patch:")}
    # Card-access Turn-1 reachability stops before attacking. Any attack text,
    # even if it moves cards/energy back to hand, belongs to a later
    # attack-readiness / board-state model and should not pollute this backlog.
    if "attack_effect" in fam_base:
        return False
    # Pure Energy movement is also board-state / attack-readiness, not access to
    # the requested card pieces.
    if fam_base and fam_base <= IGNORABLE_FAMILIES:
        return False
    if "energy_move" in fam_base and not (fam_base & {"random_draw", "deck_search_to_hand", "deck_search_to_bench", "topdeck_setup", "top_n_choose"}):
        return False
    if fam_base & TURN1_RELEVANT_FAMILIES:
        return True
    t = normalize(text)
    access_patterns = [
        r"\bdraw\b",
        r"search your deck",
        r"put (it|them|[0-9]+|up to [0-9]+) .*into your hand",
        r"put (it|them|[0-9]+|up to [0-9]+) .*onto your bench",
        r"look at the (top|bottom) [0-9]+",
        r"\battach (a|an|up to|any number of|[0-9]+)",
        r"from your hand onto your bench",
        r"top of your deck",
        r"shuffle your hand",
    ]
    if any(re.search(p, t) for p in access_patterns):
        return True
    return any(op in op_list for op in ("draw_cards", "search_deck", "look_at_cards", "look_at_top_cards", "draw_until_hand_size", "draw_until_hand_size_matches", "attach_card"))


def trust_status(effect: Dict[str, Any], families: Sequence[str]) -> str:
    support = normalize(runtime_support(effect))
    semantic = has_semantic_marker(effect)
    op_set = set(ops(effect))
    fam = set(families)

    # For card-access Turn-1 reports, attack text is always irrelevant. Do this
    # before top_n_choose so attacks like Chien-Pao Icicle Loop do not appear as
    # missing top-N executors.
    if "attack_effect" in fam:
        return "irrelevant_to_turn1_access"

    if "supporter_rule" in fam and op_set <= {"reference_global_rule"}:
        return "trusted_global_rule"
    if "conditional_comeback_guard" in fam:
        return "trusted_with_runtime_guards"
    if "not_executable_without_more_structure" in support:
        return "not_executable_without_more_structure"
    if "structured_override" in support or "structured_from_compiled_text" in support:
        # Still mark guarded families separately; they need state checks, but are usable.
        if fam & {"activated_ability", "bench_from_hand_trigger", "conditional_attach_then_draw", "requires_active_source"}:
            return "trusted_with_runtime_guards"
        return "trusted_runtime"
    if semantic:
        if fam & {"runtime_patch:semantic_ir_structured_override", "runtime_patch:generic_random_draw", "runtime_patch:conditional_attach_then_draw"}:
            return "trusted_with_runtime_guards"
        return "semantic_only_not_trusted"

    # No semantic marker, structured ops can be trusted if simple enough.
    if fam <= IGNORABLE_FAMILIES or ("attack_effect" in fam and not (fam & TURN1_RELEVANT_FAMILIES)) or ("energy_move" in fam and not (fam & TURN1_RELEVANT_FAMILIES)):
        return "irrelevant_to_turn1_access"
    if "top_n_choose" in fam:
        return "needs_family_executor_top_n_choose"
    if "conditional_attach_then_draw" in fam or "activated_ability" in fam or "bench_from_hand_trigger" in fam:
        return "needs_runtime_guard"
    if "search_deck" in op_set or "draw_cards" in op_set or "move_card" in op_set:
        return "trusted_from_compiled_ops"
    if fam <= IGNORABLE_FAMILIES:
        return "irrelevant_to_turn1_access"
    return "unknown_needs_review"


CARD_ID_RE = re.compile(r"^[a-z]{1,8}\d*-\d+[a-z]?$", flags=re.I)


def _add_deck_token(value: Any, ids: set[str], names: set[str]) -> None:
    """Add either a Pokémon TCG API card id or a normalized card name.

    Handles these common forms:
      - sv6-221
      - 3 sv6-221
      - 3,sv6-221
      - 4 Ultra Ball
      - Ultra Ball
    """
    raw = str(value or "").strip()
    if not raw:
        return
    # Strip a leading count, if present.
    m = re.match(r"^\s*\d+\s*[xX]?\s+(.+?)\s*$", raw)
    if m:
        raw = m.group(1).strip()
    # Strip CSV-ish quantity prefix if the line was not parsed as CSV.
    m = re.match(r"^\s*\d+\s*,\s*(.+?)\s*$", raw)
    if m:
        raw = m.group(1).strip()
    # If the first token is an id, prefer that. Keep remaining name as optional metadata.
    parts = raw.split()
    if parts and CARD_ID_RE.match(parts[0]):
        ids.add(parts[0])
        if len(parts) > 1:
            names.add(normalize(" ".join(parts[1:])))
        return
    if CARD_ID_RE.match(raw):
        ids.add(raw)
    else:
        names.add(normalize(raw))


def read_decklist(path: Optional[Path]) -> Tuple[set[str], set[str]]:
    ids: set[str] = set()
    names: set[str] = set()
    if not path:
        return ids, names
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ("card_id", "id", "pokemon_tcg_api_id", "api_id"):
                    _add_deck_token(row.get(key), ids, names)
                # Some CSV exports store the full line or id in a generic card column.
                for key in ("card", "entry", "line"):
                    _add_deck_token(row.get(key), ids, names)
                for key in ("name", "card_name"):
                    val = str(row.get(key) or "").strip()
                    if val:
                        names.add(normalize(val))
        return ids, names

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            _add_deck_token(line, ids, names)
    return ids, names


def first_nonempty(*vals: Any) -> str:
    for v in vals:
        if v not in (None, ""):
            return str(v)
    return ""


def card_name(card: Dict[str, Any], cid: str = "", id_to_name: Optional[Dict[str, str]] = None, effect: Optional[Dict[str, Any]] = None) -> str:
    id_to_name = id_to_name or {}
    src = (effect.get("source") or {}) if effect else {}
    return first_nonempty(
        card.get("name"),
        card.get("card_name"),
        card.get("display_name"),
        src.get("card_name"),
        id_to_name.get(cid, ""),
    )


def card_id(card: Dict[str, Any], effect: Optional[Dict[str, Any]] = None) -> str:
    if effect:
        sid = source_card_id(effect)
        if sid:
            return sid
    return first_nonempty(card.get("card_id"), card.get("id"), card.get("pokemon_tcg_api_id"))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(compiled: Any, deck_ids: set[str], deck_names: set[str], id_to_name: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    id_to_name = id_to_name or {}
    rows: List[Dict[str, Any]] = []
    for card in get_cards(compiled):
        effects = card.get("compiled_effects") or card.get("effects") or []
        if not isinstance(effects, list):
            continue
        for idx, effect in enumerate(effects):
            if not isinstance(effect, dict):
                continue
            cid = card_id(card, effect)
            cname = card_name(card, cid, id_to_name, effect)
            ename = effect_name(effect)
            text = effect_text(effect)
            op_list = ops(effect)
            families = classify_families(card, effect)
            relevant = is_turn1_relevant(families, text, op_list)
            trust = trust_status(effect, families)
            norm_cname = normalize(cname)
            norm_ename = normalize(ename)
            exact_id_match = cid in deck_ids
            name_match = norm_cname in deck_names or norm_ename in deck_names
            in_deck = exact_id_match or name_match
            match_reason = "exact_id" if exact_id_match else ("resolved_name" if name_match else "")
            rows.append({
                "card_id": cid,
                "card_name": cname,
                "effect_index": idx,
                "effect_id": effect.get("effect_id", ""),
                "effect_name": ename,
                "kind": effect.get("kind", ""),
                "section": source_section(effect),
                "ops": ",".join(op_list),
                "families": ";".join(families),
                "turn1_relevant": str(bool(relevant)).lower(),
                "trust_status": trust,
                "runtime_support": runtime_support(effect),
                "runtime_patches": ";".join(runtime_patches(effect)),
                "has_semantic_ir_marker": str(has_semantic_marker(effect)).lower(),
                "in_deck": str(bool(in_deck)).lower(),
                "deck_match_reason": match_reason,
                "exact_deck_id_match": str(bool(exact_id_match)).lower(),
                "resolved_name_match": str(bool(name_match)).lower(),
                "effect_signature": effect_signature(cid, cname, effect, text, op_list, families),
                "text": text.replace("\n", " ")[:1000],
            })
    return rows



def effect_signature(cid: str, cname: str, effect: Dict[str, Any], text: str, op_list: Sequence[str], families: Sequence[str]) -> str:
    """Signature used to collapse alternate-print noise for deck reports.

    Exact deck-id rows remain exact. For name fallback rows, different historical
    prints often share the same practical effect. We group by card name + effect
    text + operation/family shape so the current-deck report shows one usable
    representative instead of every historical alternate print.
    """
    key = "|".join([
        normalize(cname),
        normalize(effect_name(effect)),
        normalize(effect.get("kind")),
        normalize(source_section(effect)),
        normalize(",".join(op_list)),
        normalize(";".join(families)),
        normalize(text)[:500],
    ])
    return str(abs(hash(key)))


def representative_deck_rows(rows: List[Dict[str, Any]], deck_ids: set[str], deck_names: set[str]) -> List[Dict[str, Any]]:
    """Return a clean deck report: exact id rows plus one representative per fallback effect.

    v0.3 deliberately matched by resolved name to handle deduplicated compiled
    entries, but that pulled in all historical Ultra Ball / Energy Switch prints.
    v0.4 keeps a full expanded-by-name CSV, while this function produces the
    default current-deck audit: exact id rows when present; otherwise deduped
    resolved-name representatives.
    """
    if not (deck_ids or deck_names):
        return []

    exact = [r for r in rows if r.get("exact_deck_id_match") == "true"]
    exact_names = {normalize(r.get("card_name")) for r in exact if r.get("card_name")}
    out: List[Dict[str, Any]] = list(exact)
    seen = {(r.get("card_id"), r.get("effect_id"), r.get("effect_signature")) for r in out}

    fallback_candidates = [
        r for r in rows
        if r.get("resolved_name_match") == "true"
        and r.get("exact_deck_id_match") != "true"
        # If we already have exact rows for this name, keep the alternates only
        # in the expanded report, not the clean deck audit.
        and normalize(r.get("card_name")) not in exact_names
    ]

    # Deduplicate by card name + effect signature, preferring rows with names,
    # trusted statuses, and non-empty runtime metadata.
    best: Dict[tuple, Dict[str, Any]] = {}
    trust_rank = {
        "trusted_runtime": 0,
        "trusted_with_runtime_guards": 1,
        "trusted_global_rule": 2,
        "trusted_from_compiled_ops": 3,
        "needs_family_executor_top_n_choose": 4,
        "needs_runtime_guard": 5,
        "unknown_needs_review": 6,
        "not_executable_without_more_structure": 7,
        "semantic_only_not_trusted": 8,
        "irrelevant_to_turn1_access": 9,
    }
    for r in fallback_candidates:
        k = (normalize(r.get("card_name")), r.get("effect_signature"))
        score = (trust_rank.get(str(r.get("trust_status")), 99), 0 if r.get("runtime_support") else 1, str(r.get("card_id")))
        if k not in best:
            best[k] = r
        else:
            old = best[k]
            old_score = (trust_rank.get(str(old.get("trust_status")), 99), 0 if old.get("runtime_support") else 1, str(old.get("card_id")))
            if score < old_score:
                best[k] = r
    for r in best.values():
        key = (r.get("card_id"), r.get("effect_id"), r.get("effect_signature"))
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Turn-1 runtime effect-family coverage from compiled Pokémon TCG cards.")
    ap.add_argument("--compiled", required=True, help="Path to compiled_cards_all*.json")
    ap.add_argument("--decklist", default=None, help="Optional decklist file to mark current-deck effects")
    ap.add_argument("--cards-csv", default="data/all_cards.csv", help="Optional all_cards.csv used to resolve alternate-print ids to card names")
    ap.add_argument("--outdir", default="data/reports/runtime_coverage", help="Output directory")
    ap.add_argument("--prefix", default="", help="Optional filename prefix")
    ap.add_argument("--max-examples", type=int, default=50, help="Examples per summary bucket in JSON")
    args = ap.parse_args()

    compiled_path = Path(args.compiled)
    outdir = Path(args.outdir)
    prefix = (args.prefix + "_") if args.prefix else ""

    compiled = load_json(compiled_path)
    id_to_name = load_card_name_map(Path(args.cards_csv) if args.cards_csv else None)
    deck_ids, deck_names = read_decklist(Path(args.decklist) if args.decklist else None)
    # Resolve deck ids to card names so alternate prints in the compiled corpus still match.
    for did in list(deck_ids):
        nm = id_to_name.get(did)
        if nm:
            deck_names.add(normalize(nm))
    rows = build_rows(compiled, deck_ids, deck_names, id_to_name)

    all_csv = outdir / f"{prefix}all_cards_runtime_families.csv"
    unsupported_csv = outdir / f"{prefix}turn1_relevant_unsupported.csv"
    deck_csv = outdir / f"{prefix}current_deck_runtime_audit.csv"
    expanded_deck_csv = outdir / f"{prefix}current_deck_runtime_audit_expanded_by_name.csv"
    family_csv = outdir / f"{prefix}family_summary.csv"
    summary_json = outdir / f"{prefix}summary.json"

    unsupported_statuses = {
        "semantic_only_not_trusted",
        "not_executable_without_more_structure",
        "needs_family_executor_top_n_choose",
        "needs_runtime_guard",
        "unknown_needs_review",
    }
    unsupported = [r for r in rows if r["turn1_relevant"] == "true" and r["trust_status"] in unsupported_statuses]
    expanded_deck_rows = [r for r in rows if r["in_deck"] == "true"] if (deck_ids or deck_names) else []
    deck_rows = representative_deck_rows(rows, deck_ids, deck_names) if (deck_ids or deck_names) else []

    family_counter = Counter()
    family_trust = Counter()
    for r in rows:
        for fam in str(r["families"]).split(";"):
            family_counter[fam] += 1
            family_trust[(fam, r["trust_status"])] += 1
    family_rows: List[Dict[str, Any]] = []
    for fam, count in family_counter.most_common():
        fam_rows = [r for r in rows if fam in str(r["families"]).split(";")]
        rel_count = sum(1 for r in fam_rows if r["turn1_relevant"] == "true")
        unsupported_count = sum(1 for r in fam_rows if r["trust_status"] in unsupported_statuses)
        top_trust = Counter(r["trust_status"] for r in fam_rows).most_common(5)
        family_rows.append({
            "family": fam,
            "effect_count": count,
            "turn1_relevant_count": rel_count,
            "unsupported_count": unsupported_count,
            "top_trust_statuses": "; ".join(f"{k}:{v}" for k, v in top_trust),
        })

    fields = [
        "card_id", "card_name", "effect_index", "effect_id", "effect_name", "kind", "section",
        "ops", "families", "turn1_relevant", "trust_status", "runtime_support", "runtime_patches",
        "has_semantic_ir_marker", "in_deck", "deck_match_reason", "exact_deck_id_match", "resolved_name_match", "effect_signature", "text",
    ]
    write_csv(all_csv, rows, fields)
    write_csv(unsupported_csv, unsupported, fields)
    if deck_ids or deck_names:
        write_csv(deck_csv, deck_rows, fields)
        write_csv(expanded_deck_csv, expanded_deck_rows, fields)
    write_csv(family_csv, family_rows, ["family", "effect_count", "turn1_relevant_count", "unsupported_count", "top_trust_statuses"])

    trust_counts = Counter(r["trust_status"] for r in rows)
    relevance_counts = Counter(r["turn1_relevant"] for r in rows)
    deck_trust_counts = Counter(r["trust_status"] for r in deck_rows)
    summary = {
        "version": VERSION,
        "compiled": str(compiled_path),
        "cards_csv": args.cards_csv,
        "decklist": args.decklist,
        "deck_ids_parsed": len(deck_ids),
        "deck_names_resolved": len(deck_names),
        "effects_total": len(rows),
        "turn1_relevant_effects": relevance_counts.get("true", 0),
        "unsupported_turn1_relevant_effects": len(unsupported),
        "deck_effects": len(deck_rows),
        "deck_effects_expanded_by_name": len(expanded_deck_rows),
        "trust_counts": dict(trust_counts.most_common()),
        "deck_trust_counts": dict(deck_trust_counts.most_common()),
        "family_counts_top": dict(family_counter.most_common(30)),
        "outputs": {
            "all_cards_runtime_families_csv": str(all_csv),
            "turn1_relevant_unsupported_csv": str(unsupported_csv),
            "current_deck_runtime_audit_csv": str(deck_csv) if (deck_ids or deck_names) else None,
            "current_deck_runtime_audit_expanded_by_name_csv": str(expanded_deck_csv) if (deck_ids or deck_names) else None,
            "family_summary_csv": str(family_csv),
            "summary_json": str(summary_json),
        },
        "unsupported_examples": unsupported[: args.max_examples],
        "deck_examples": deck_rows[: args.max_examples],
        "deck_examples_expanded_by_name": expanded_deck_rows[: min(args.max_examples, 20)],
    }
    dump_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
