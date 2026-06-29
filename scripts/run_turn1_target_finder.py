# TURN1_DIRECT_TARGETFINDER_V40_CACHE_V54
from __future__ import annotations

"""
Turn 1 target-finding simulator for a Pokémon TCG deck.

Version: v0.15 — deduplicates compiled/source abilities and fixes ability line labels from nested source names.

Purpose
-------
Estimate P(find target card X by the end of turn 1) under simple optimal play.

This is NOT a full game engine. It is a scenario solver for the narrow question:
"Given my opening hand, prizes, draw-for-turn, first/second turn restrictions, and
compiled draw/search/look/reorder effects plus Pokémon Abilities, can optimal play find X?"

The policy is conservative and transparent:
  1. Success if target is already in hand.
  2. Put 6 prizes aside after opening hand.
  3. Draw for turn by default.
  4. Play legal Trainer effects from hand in greedy best-first order.
  5. Direct search for the target beats draw effects.
  6. If direct target search is unavailable, draw effects are preferred.
  7. Optional chain search can fetch another playable enabler from deck.
  8. Deck-specific combo support includes Ultra Ball, Cyrano, Ciphermaniac + Run Errand,
     Meowth ex -> Supporter, Lillie's Determination draw, plus a generic compiled-ability executor.
  9. v0.7 adds a line-audit report that checks whether combo lines have the required
     setup/action evidence in the simulated log.
 10. v0.8 imports src/probability.py and reports exact legal-hand baselines beside the simulated action phase.
 11. v0.9 writes the large JSON/CSV reports to the user Downloads folder by default and prints only a compact summary.
 12. v0.10 adds a generic Pokémon Ability layer: Basic Pokémon with useful abilities can be benched, active/bench abilities can be used once, and compiled ability draw/search/look/reorder effects are evaluated for every Pokémon, not only deck-specific hand-coded abilities.
 13. v0.12 expands that layer for conditional/costed abilities: requirements such as "if you have X in play", costs such as discarding Basic typed Energy from hand, and once-per-turn ability-family limits are now interpreted generically.
 14. v0.13 adds search-for-ability-requirement chains. If an ability needs another Basic Pokémon in play, the policy can use a legal search card such as Ultra Ball or Fighting Gong to fetch that requirement, bench it, then use the ability. This covers lines like Lunatone in play + Ultra Ball -> Solrock -> Lunar Cycle -> draw 3.
 15. v0.14 adds source-text fallback abilities. If an ability is present in sources.abilities / raw API data but absent from compiled_effects, the script synthesizes a conservative draw/search/look effect from the printed ability text. This fixes ability blindspots such as Lunar Cycle when the compiler did not emit an ability effect.
 16. v0.15 fixes ability identity/labels now that compiler v0.9 can emit Lunar Cycle directly: nested source.name is used as the ability name, compiled/source duplicate abilities are deduplicated by printed text, and once-per-turn ability usage keys are stable.

The output logs the lines it used so you can audit what "optimal" meant.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from tcgsim import load_compiled_cards, filter_complete_cards
except Exception:  # pragma: no cover - lets the file still show a useful error if copied alone
    load_compiled_cards = None
    filter_complete_cards = None

try:
    from probability import (
        p_no_basic_opening_7,
        p_card_in_legal_opening_7,
        p_card_in_hand_after_turn_draw_given_legal_opening,
        p_at_least_one_prized_after_legal_hand,
        expected_prized_after_legal_hand,
        p_all_copies_prized_after_legal_hand,
        p_still_prized_after_x_prizes_taken_after_legal_hand,
        p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand,
        mulligan_distribution_exact,
    )
except Exception:  # pragma: no cover
    p_no_basic_opening_7 = None
    p_card_in_legal_opening_7 = None
    p_card_in_hand_after_turn_draw_given_legal_opening = None
    p_at_least_one_prized_after_legal_hand = None
    expected_prized_after_legal_hand = None
    p_all_copies_prized_after_legal_hand = None
    p_still_prized_after_x_prizes_taken_after_legal_hand = None
    p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand = None
    mulligan_distribution_exact = None


# -----------------------------
# Generic helpers
# -----------------------------


def norm(s: Any) -> str:
    text = unicodedata.normalize("NFKD", str(s or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def ci95(successes: int, n: int) -> Dict[str, float]:
    if n <= 0:
        return {"low": 0.0, "high": 0.0}
    p = successes / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return {"low": pct(max(0.0, p - 1.96 * se)), "high": pct(min(1.0, p + 1.96 * se))}


def ncr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def hypergeom_at_least_one(deck_size: int, copies: int, draws: int) -> float:
    if deck_size <= 0 or draws <= 0 or copies <= 0:
        return 0.0
    if draws > deck_size:
        draws = deck_size
    return 1.0 - (ncr(deck_size - copies, draws) / ncr(deck_size, draws))


def hypergeom_zero(deck_size: int, copies: int, draws: int) -> float:
    if deck_size <= 0 or draws <= 0:
        return 1.0
    if copies <= 0:
        return 1.0
    if draws > deck_size:
        draws = deck_size
    return ncr(deck_size - copies, draws) / ncr(deck_size, draws)


def _df_to_records_safe(df: Any) -> List[Dict[str, Any]]:
    """Convert a pandas DataFrame to JSON-safe records without making pandas a hard dependency here."""
    if df is None:
        return []
    try:
        records = df.to_dict(orient="records")
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = round(v, 10)
            else:
                clean[k] = v
        out.append(clean)
    return out


def build_exact_probability_baselines(
    deck: List[Dict[str, Any]],
    target_norm: str,
    hand_size: int = 7,
    prize_count: int = 6,
    draw_for_turn: bool = True,
    max_mulligans: int = 6,
) -> Dict[str, Any]:
    """Use src/probability.py for exact opening/draw/prize calculations.

    These are the analytic calculations from the app's probability module. They are
    exact for the pre-action part of the game: legal opening hand, mulligans,
    natural turn draw, and prize probabilities after a kept legal hand. The
    simulator should then estimate only the conditional value of actions such as
    Ultra Ball, Cyrano, Run Errand, etc.
    """
    deck_size = len(deck)
    target_copies = sum(1 for c in deck if target_matches(c, target_norm))
    basic_count = sum(1 for c in deck if is_basic_pokemon(c))
    target_cards = [c for c in deck if target_matches(c, target_norm)]
    target_is_basic = bool(target_cards and is_basic_pokemon(target_cards[0]))

    if p_no_basic_opening_7 is None:
        return {
            "available": False,
            "reason": "Could not import src/probability.py. The script will still report simulation-only estimates.",
        }

    q_no_basic = p_no_basic_opening_7(deck_size, basic_count)
    p_open = p_card_in_legal_opening_7(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=target_copies,
        card_is_basic=target_is_basic,
    )
    if draw_for_turn:
        p_by_draw = p_card_in_hand_after_turn_draw_given_legal_opening(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=target_copies,
            card_is_basic=target_is_basic,
        )
    else:
        p_by_draw = p_open
    p_draw_increment = max(0.0, p_by_draw - p_open)

    prize_rows = []
    for prizes_taken in range(prize_count + 1):
        prize_rows.append({
            "prizes_taken": prizes_taken,
            "at_least_one_copy_still_prized_percent": pct(p_still_prized_after_x_prizes_taken_after_legal_hand(
                deck_size=deck_size,
                basic_count=basic_count,
                card_count=target_copies,
                card_is_basic=target_is_basic,
                prizes_taken=prizes_taken,
                starting_prize_count=prize_count,
                hand_size=hand_size,
            )),
            "all_copies_still_prized_percent": pct(p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand(
                deck_size=deck_size,
                basic_count=basic_count,
                card_count=target_copies,
                card_is_basic=target_is_basic,
                prizes_taken=prizes_taken,
                starting_prize_count=prize_count,
                hand_size=hand_size,
            )),
        })

    mulligan_rows = []
    if mulligan_distribution_exact is not None:
        mulligan_rows = _df_to_records_safe(mulligan_distribution_exact(q_no_basic, max_mulligans=max_mulligans))
        for row in mulligan_rows:
            if "probability" in row:
                row["percent"] = pct(float(row["probability"]))

    return {
        "available": True,
        "source": "src/probability.py",
        "conditioning": "All opening-hand, draw-for-turn, mulligan, and prize probabilities are conditioned on keeping a legal opening hand with at least one Basic Pokémon.",
        "deck_size": deck_size,
        "basic_pokemon": basic_count,
        "target_copies": target_copies,
        "target_is_basic_pokemon": target_is_basic,
        "no_basic_opening_hand_percent": pct(q_no_basic),
        "legal_opening_hand_percent": pct(1.0 - q_no_basic),
        "opening_hand_has_target_percent": pct(p_open),
        "natural_draw_by_turn_1_has_target_percent": pct(p_by_draw),
        "draw_for_turn_increment_percent": pct(p_draw_increment),
        "at_least_one_target_prized_after_legal_hand_percent": pct(p_at_least_one_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=target_copies,
            card_is_basic=target_is_basic,
            prize_count=prize_count,
            hand_size=hand_size,
        )),
        "expected_target_copies_prized_after_legal_hand": round(expected_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=target_copies,
            card_is_basic=target_is_basic,
            prize_count=prize_count,
            hand_size=hand_size,
        ), 6),
        "all_target_copies_prized_after_legal_hand_percent": pct(p_all_copies_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=target_copies,
            card_is_basic=target_is_basic,
            prize_count=prize_count,
            hand_size=hand_size,
        )),
        "mulligan_distribution_exact": mulligan_rows,
        "remaining_prize_probabilities_after_prizes_taken": prize_rows,
    }


def add_exact_plus_simulation_fields(scenarios: List[Dict[str, Any]], exact: Dict[str, Any]) -> None:
    """Blend exact pre-action probabilities with simulated conditional action value."""
    if not exact.get("available"):
        return
    p_open = float(exact["opening_hand_has_target_percent"]) / 100.0
    p_by_draw = float(exact["natural_draw_by_turn_1_has_target_percent"]) / 100.0
    p_draw_increment = max(0.0, p_by_draw - p_open)
    p_not_by_draw = max(0.0, 1.0 - p_by_draw)

    for scenario in scenarios:
        sm = scenario.get("summary", {})
        n = int(sm.get("trials") or 0)
        opening_count = int(sm.get("found_in_opening_hand", {}).get("successes") or 0)
        draw_count = int(sm.get("found_on_draw_for_turn", {}).get("successes") or 0)
        action_count = int(sm.get("found_after_actions", {}).get("successes") or 0)
        not_seen_after_draw_count = max(0, n - opening_count - draw_count)
        conditional_action_rate = (action_count / not_seen_after_draw_count) if not_seen_after_draw_count else 0.0
        final_probability = p_by_draw + p_not_by_draw * conditional_action_rate

        line_rows = []
        for row in sm.get("top_success_lines", []) or []:
            count = int(row.get("count") or 0)
            conditional_line_rate = (count / not_seen_after_draw_count) if not_seen_after_draw_count else 0.0
            line_rows.append({
                **row,
                "conditional_on_not_seen_after_draw_percent": pct(conditional_line_rate),
                "exact_weighted_percent_of_trials": pct(p_not_by_draw * conditional_line_rate),
            })

        scenario["exact_plus_simulation"] = {
            "method": "Exact opening/draw from src/probability.py + simulated conditional action success from this script.",
            "exact_opening_hand_percent": pct(p_open),
            "exact_draw_for_turn_increment_percent": pct(p_draw_increment),
            "exact_seen_by_draw_for_turn_percent": pct(p_by_draw),
            "simulated_action_success_given_not_seen_after_draw_percent": pct(conditional_action_rate),
            "exact_weighted_action_increment_percent": pct(p_not_by_draw * conditional_action_rate),
            "final_exact_plus_sim_percent": pct(final_probability),
            "line_contributions": line_rows,
        }


def card_identity(card: Dict[str, Any]) -> Dict[str, Any]:
    return card.get("identity", {}) or {}


def card_id(card: Dict[str, Any]) -> str:
    return str(card.get("card_id") or card.get("representative_card_id") or card_identity(card).get("card_id") or "unknown")


def card_name(card: Dict[str, Any]) -> str:
    i = card_identity(card)
    return str(i.get("name") or i.get("canonical_name") or card.get("name") or card_id(card))


def card_supertype(card: Dict[str, Any]) -> str:
    return str(card_identity(card).get("supertype") or "")


def card_subtypes(card: Dict[str, Any]) -> List[str]:
    return list(card_identity(card).get("subtypes") or [])


def card_types(card: Dict[str, Any]) -> List[str]:
    return list(card_identity(card).get("types") or [])


def is_basic_pokemon(card: Dict[str, Any]) -> bool:
    return card_supertype(card) == "Pokémon" and "Basic" in card_subtypes(card)


def is_energy(card: Dict[str, Any]) -> bool:
    return card_supertype(card) == "Energy"


def is_trainer(card: Dict[str, Any]) -> bool:
    return card_supertype(card) == "Trainer"


def is_supporter(card: Dict[str, Any]) -> bool:
    return "Supporter" in card_subtypes(card) or bool((card.get("gameplay", {}) or {}).get("trainer", {}).get("counts_as_supporter"))


def is_item_like(card: Dict[str, Any]) -> bool:
    subs = set(card_subtypes(card))
    return bool(subs & {"Item", "Pokémon Tool", "Tool", "Stadium"}) or bool((card.get("gameplay", {}) or {}).get("trainer", {}).get("counts_as_item"))


def is_named(card: Dict[str, Any], name: str) -> bool:
    return norm(card_name(card)) == norm(name)


def is_ultra_ball(card: Dict[str, Any]) -> bool:
    return is_named(card, "Ultra Ball")


def is_cyrano(card: Dict[str, Any]) -> bool:
    return is_named(card, "Cyrano")


def is_ciphermaniac(card: Dict[str, Any]) -> bool:
    return is_named(card, "Ciphermaniac's Codebreaking")


def is_lillies_determination(card: Dict[str, Any]) -> bool:
    return is_named(card, "Lillie's Determination")


def is_crispin(card: Dict[str, Any]) -> bool:
    return is_named(card, "Crispin")


def is_mega_kangaskhan_ex(card: Dict[str, Any]) -> bool:
    return is_named(card, "Mega Kangaskhan ex")


def is_meowth_ex(card: Dict[str, Any]) -> bool:
    return is_named(card, "Meowth ex")


def is_pokemon_ex(card: Dict[str, Any]) -> bool:
    return card_supertype(card) == "Pokémon" and "ex" in set(card_subtypes(card))


def target_is_pokemon_ex_in_pool(target_norm: str, pool: Sequence[Dict[str, Any]]) -> bool:
    return any(target_matches(c, target_norm) and is_pokemon_ex(c) for c in pool)


def find_card_in_zone(zone: Sequence[Dict[str, Any]], pred) -> Optional[Dict[str, Any]]:
    for c in zone:
        if pred(c):
            return c
    return None


def has_card_in_zone(zone: Sequence[Dict[str, Any]], pred) -> bool:
    return find_card_in_zone(zone, pred) is not None


def target_matches(card: Dict[str, Any], target_norm: str) -> bool:
    if not target_norm:
        return False
    names = [
        card_name(card),
        card_identity(card).get("canonical_name"),
        card.get("card_id"),
        card.get("representative_card_id"),
    ]
    for p in card.get("same_effect_printings", []) or []:
        if isinstance(p, dict):
            names.extend([p.get("name"), p.get("card_id"), p.get("id")])
    for x in names:
        nx = norm(x)
        if nx and (nx == target_norm or target_norm in nx):
            return True
    return False


def amount_value(amount: Any, default: int = 0, counts: Optional[Dict[str, int]] = None, coin_heads: int = 0) -> int:
    counts = counts or {}
    if amount is None:
        return default
    if isinstance(amount, bool):
        return int(amount)
    if isinstance(amount, int):
        return amount
    if isinstance(amount, float):
        return int(amount)
    if isinstance(amount, str):
        m = re.search(r"\d+", amount)
        return int(m.group(0)) if m else default
    if isinstance(amount, dict):
        if "value" in amount and isinstance(amount["value"], (int, float)):
            return int(amount["value"])
        if "base" in amount and isinstance(amount["base"], (int, float)):
            return int(amount["base"])
        if amount.get("mode") == "count_ref":
            return int(counts.get(str(amount.get("count_id")), default))
        if amount.get("mode") == "coin_heads":
            return int(coin_heads)
        if amount.get("mode") == "per_coin_heads":
            return int(amount.get("value", default)) * int(coin_heads)
        if "printed" in amount:
            return amount_value(amount.get("printed"), default=default, counts=counts, coin_heads=coin_heads)
    return default


# -----------------------------
# Decklist resolution
# -----------------------------


def parse_decklist(path: str) -> List[Tuple[int, str]]:
    """Parse common decklist formats.

    Supported:
      - TXT: lines like "4 Professor's Research" or "Professor's Research x4"
      - CSV: count/name columns, or two columns count,name
      - JSON: {"Card Name": 4} or [{"name": ..., "count": ...}]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return [(int(v), str(k)) for k, v in payload.items()]
        if isinstance(payload, list):
            out: List[Tuple[int, str]] = []
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = row.get("name") or row.get("card") or row.get("card_name")
                count = row.get("count") or row.get("qty") or row.get("quantity") or row.get("copies")
                if name and count:
                    out.append((int(float(count)), str(name).strip()))
            return out
        raise ValueError(f"Unsupported JSON decklist structure: {path}")

    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            if has_header:
                reader = csv.DictReader(f)
                out: List[Tuple[int, str]] = []
                for row in reader:
                    lower = {norm(k): v for k, v in row.items()}
                    count = lower.get("count") or lower.get("qty") or lower.get("quantity") or lower.get("copies")
                    name = lower.get("name") or lower.get("card") or lower.get("card name") or lower.get("cardname")
                    if count and name:
                        out.append((int(float(str(count).strip())), str(name).strip()))
                return out
            reader = csv.reader(f)
            out = []
            for row in reader:
                if len(row) >= 2 and str(row[0]).strip().isdigit():
                    out.append((int(row[0]), str(row[1]).strip()))
            return out

    out: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            if norm(line) in {"pokemon", "trainer", "trainers", "energy", "energies"}:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                out.append((int(m.group(1)), m.group(2).strip()))
                continue
            m = re.match(r"^(.+?)\s+[xX](\d+)$", line)
            if m:
                out.append((int(m.group(2)), m.group(1).strip()))
                continue
    return out


def build_name_index(cards: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cards:
        names = {
            card_name(c),
            card_identity(c).get("canonical_name"),
            c.get("name"),
            c.get("card_id"),
            c.get("representative_card_id"),
        }
        for p in c.get("same_effect_printings", []) or []:
            if isinstance(p, dict):
                names.update({p.get("name"), p.get("card_id"), p.get("id")})
        for name in names:
            key = norm(name)
            if key:
                index[key].append(c)
    return index




def basic_energy_proxy_from_request(requested: str) -> Optional[Dict[str, Any]]:
    """Create a lightweight Basic Energy card when an Energy printing is absent from compiled JSON.

    The compiler may omit vanilla Basic Energy printings because they have no card text. For
    probability questions, those cards still need to occupy slots in the 60-card deck. This
    function turns obvious Basic Energy decklist entries into safe no-effect proxy cards.
    """
    raw = str(requested or "").strip()
    n = norm(raw)
    raw_upper = raw.upper()

    # Type hints from common Pokémon decklist symbols and names.
    symbol_map = {
        "{G}": "Grass",
        "{R}": "Fire",
        "{W}": "Water",
        "{L}": "Lightning",
        "{P}": "Psychic",
        "{F}": "Fighting",
        "{D}": "Darkness",
        "{M}": "Metal",
    }
    energy_type: Optional[str] = None
    for sym, typ in symbol_map.items():
        if sym in raw_upper:
            energy_type = typ
            break

    if energy_type is None:
        for typ in ["Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal"]:
            if norm(typ) in n:
                energy_type = typ
                break

    # Handle the mistaken/short IDs commonly used while testing this project, plus real SVE IDs.
    # The v0.1 decklist used sve-5/4/6/1/3 to stand for MEE 5/4/6/1/3 from the pasted decklist.
    id_hint_map = {
        "sve 1": "Grass",
        "sve 3": "Water",
        "sve 4": "Lightning",
        "sve 5": "Psychic",
        "sve 6": "Fighting",
        "sve 10": "Fire",
        "sve 11": "Water",
        "sve 12": "Lightning",
        "sve 13": "Psychic",
        "sve 14": "Fighting",
        "sve 15": "Darkness",
        "sve 16": "Metal",
        "mee 1": "Grass",
        "mee 3": "Water",
        "mee 4": "Lightning",
        "mee 5": "Psychic",
        "mee 6": "Fighting",
    }
    if energy_type is None:
        energy_type = id_hint_map.get(n)

    looks_like_energy = "energy" in n or n in id_hint_map or n.startswith("sve ") or n.startswith("mee ")
    if not energy_type or not looks_like_energy:
        return None

    name = f"Basic {energy_type} Energy"
    proxy_id = f"proxy-energy-{norm(energy_type).replace(' ', '-')}-{re.sub(r'[^a-z0-9]+', '-', n).strip('-')}"
    return {
        "card_id": proxy_id,
        "representative_card_id": proxy_id,
        "identity": {
            "card_id": proxy_id,
            "name": name,
            "canonical_name": name,
            "supertype": "Energy",
            "subtypes": ["Basic"],
            "types": [energy_type],
        },
        "gameplay": {},
        "compiled_effects": [],
        "parser_status": "proxy_basic_energy",
        "source": {"proxy_for_decklist_entry": raw},
        "same_effect_printings": [
            {"card_id": raw, "id": raw, "name": name}
        ],
    }

def resolve_decklist(decklist: List[Tuple[int, str]], cards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index = build_name_index(cards)
    all_keys = list(index.keys())
    deck: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for count, requested in decklist:
        key = norm(requested)
        candidates = index.get(key) or []
        if not candidates:
            hits = [k for k in all_keys if key and key in k]
            unique: List[Dict[str, Any]] = []
            seen = set()
            for h in hits:
                for c in index[h]:
                    cid = card_id(c)
                    if cid not in seen:
                        unique.append(c)
                        seen.add(cid)
            if len(unique) == 1:
                candidates = unique
        if not candidates:
            proxy = basic_energy_proxy_from_request(requested)
            if proxy is not None:
                for _ in range(int(count)):
                    deck.append(proxy)
                continue
            unresolved.append({"requested_name": requested, "count": count})
            continue
        chosen = candidates[0]
        for _ in range(int(count)):
            deck.append(chosen)
    return deck, unresolved


# -----------------------------
# Compiled-effect helpers
# -----------------------------


def iter_effects(card: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for eff in card.get("compiled_effects", []) or []:
        if isinstance(eff, dict):
            yield eff


def iter_steps(effect_or_steps: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(effect_or_steps, dict):
        steps = effect_or_steps.get("steps", [])
    else:
        steps = effect_or_steps
    if not isinstance(steps, list):
        return
    for s in steps:
        if isinstance(s, dict):
            yield s


def flatten_steps(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "op" in obj:
            yield obj
        for key in ("steps", "then", "else", "if_true", "if_false", "heads", "tails", "yes", "no", "branches"):
            val = obj.get(key)
            if isinstance(val, list):
                for x in val:
                    yield from flatten_steps(x)
            elif isinstance(val, dict):
                yield from flatten_steps(val)
    elif isinstance(obj, list):
        for x in obj:
            yield from flatten_steps(x)


def meaningful_steps(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for eff in iter_effects(card):
        for step in iter_steps(eff):
            op = step.get("op")
            if op == "reference_global_rule":
                continue
            out.append(step)
    return out


def effect_is_trivial_rule(effect: Dict[str, Any]) -> bool:
    steps = list(iter_steps(effect))
    if not steps:
        return True
    return all(s.get("op") == "reference_global_rule" for s in steps)


def card_has_play_effect(card: Dict[str, Any]) -> bool:
    return bool(meaningful_steps(card))


def step_text(step: Dict[str, Any]) -> str:
    return str(step.get("source_text") or step.get("text") or "")


def extract_filter(step: Dict[str, Any]) -> Dict[str, Any]:
    f = step.get("filter") or step.get("card_filter") or {}
    if isinstance(f, dict) and "card" in f and isinstance(f["card"], dict):
        return f["card"]
    return f if isinstance(f, dict) else {}


def filter_text_blob(filt: Dict[str, Any]) -> str:
    """Flatten a compiled card filter into searchable text.

    Some compiler outputs keep the important restriction only as raw_text /
    source_text. The all-card simulator must not treat those vague dictionaries
    as "any card". For example, Fighting Gong may compile with raw text but
    no structured filter; it should only find Basic Fighting Energy or Basic
    Fighting Pokemon, not Trainers.
    """
    try:
        return norm(json.dumps(filt, ensure_ascii=False))
    except Exception:
        return norm(str(filt))


def is_basic_fighting_energy(card: Dict[str, Any]) -> bool:
    if card_supertype(card) != "Energy":
        return False
    name_n = norm(card_name(card))
    types = {norm(x) for x in card_types(card)}
    return ("fighting" in name_n) or ("fighting" in types)


def is_basic_fighting_pokemon(card: Dict[str, Any]) -> bool:
    if not is_basic_pokemon(card):
        return False
    types = {norm(x) for x in card_types(card)}
    return "fighting" in types


# ---------------------------------------------------------------------
# TURN1_STRICT_SEARCH_FILTERS_V37
# ---------------------------------------------------------------------
# Root fix:
# Search filters must enforce the actual target class from the card text.
#
# Examples:
# - Buddy-Buddy Poffin: Basic Pokemon with 70 HP or less only.
# - Irida: Water Pokemon OR Item only.
# - Shivery Chill: Basic Water Energy only.
#
# This replaces the old permissive behavior where raw-text filters could
# accidentally match any target card.

def _turn1_v37_norm(value):
    try:
        return norm(value)
    except Exception:
        import re as _re
        import unicodedata as _unicodedata

        s = str(value or "")
        s = _unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not _unicodedata.combining(ch))
        s = s.lower().replace("’", "'").replace("`", "'")
        s = _re.sub(r"\s+", " ", s)
        return s.strip()


def _turn1_v37_card_supertype_norm(card):
    try:
        return _turn1_v37_norm(card_supertype(card))
    except Exception:
        return ""


def _turn1_v37_card_name_norm(card):
    try:
        return _turn1_v37_norm(card_name(card))
    except Exception:
        return ""


def _turn1_v37_card_subtypes_norm(card):
    try:
        return {_turn1_v37_norm(x) for x in card_subtypes(card)}
    except Exception:
        return set()


def _turn1_v37_card_types_norm(card):
    try:
        return {_turn1_v37_norm(x) for x in card_types(card)}
    except Exception:
        return set()


def _turn1_v37_is_pokemon(card):
    return _turn1_v37_card_supertype_norm(card) == "pokemon"


def _turn1_v37_is_trainer(card):
    return _turn1_v37_card_supertype_norm(card) == "trainer"


def _turn1_v37_is_energy(card):
    try:
        return is_energy(card)
    except Exception:
        return _turn1_v37_card_supertype_norm(card) == "energy"


def _turn1_v37_is_basic_pokemon(card):
    try:
        return is_basic_pokemon(card)
    except Exception:
        return _turn1_v37_is_pokemon(card) and "basic" in _turn1_v37_card_subtypes_norm(card)


def _turn1_v37_is_basic_energy(card):
    if not _turn1_v37_is_energy(card):
        return False

    name_n = _turn1_v37_card_name_norm(card)
    subtypes = _turn1_v37_card_subtypes_norm(card)

    return "basic" in subtypes or name_n.startswith("basic ")


def _turn1_v37_card_hp(card):
    candidates = []

    if isinstance(card, dict):
        candidates.append(card.get("hp"))
        candidates.append(card.get("raw_hp"))

        ident = card.get("identity") or {}
        gameplay = card.get("gameplay") or {}
        raw_card = card.get("raw_card") or card.get("source") or {}

        if isinstance(ident, dict):
            candidates.append(ident.get("hp"))

        if isinstance(gameplay, dict):
            candidates.append(gameplay.get("hp"))

        if isinstance(raw_card, dict):
            candidates.append(raw_card.get("hp"))

    for value in candidates:
        if value is None:
            continue

        m = re.search(r"\d+", str(value))

        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

    return None


_TURN1_V37_TYPE_SYNONYMS = {
    "grass": ["grass", "g"],
    "fire": ["fire", "r"],
    "water": ["water", "w"],
    "lightning": ["lightning", "electric", "l"],
    "psychic": ["psychic", "p"],
    "fighting": ["fighting", "f"],
    "darkness": ["darkness", "dark", "d"],
    "metal": ["metal", "steel", "m"],
    "colorless": ["colorless", "c"],
}


def _turn1_v37_card_has_type(card, typ):
    typ_n = _turn1_v37_norm(typ)
    name_n = _turn1_v37_card_name_norm(card)
    types = _turn1_v37_card_types_norm(card)

    return typ_n in types or typ_n in name_n


def _turn1_v37_has_typed_phrase(text, typ, suffix):
    text = _turn1_v37_norm(text)
    synonyms = _TURN1_V37_TYPE_SYNONYMS.get(typ, [typ])

    for syn in synonyms:
        syn = _turn1_v37_norm(syn)

        if f"{syn} {suffix}" in text:
            return True

        # Handles normalized "{W} Energy" as "w energy".
        if len(syn) == 1 and f"{syn} {suffix}" in text:
            return True

    return False


def _turn1_v37_hp_limit_from_text(text):
    text = _turn1_v37_norm(text)
    m = re.search(r"(\d+)\s*hp\s*or\s*less", text)

    if not m:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None


def _turn1_v37_hp_ok(card, text):
    limit = _turn1_v37_hp_limit_from_text(text)

    if limit is None:
        return True

    hp = _turn1_v37_card_hp(card)

    if hp is None:
        return False

    return hp <= limit


def _turn1_v37_is_pokemon_ex(card):
    if not _turn1_v37_is_pokemon(card):
        return False

    name_n = _turn1_v37_card_name_norm(card)
    subtypes = _turn1_v37_card_subtypes_norm(card)

    return " ex" in f" {name_n}" or "ex" in subtypes


def _turn1_v37_trainer_kind(card, kind):
    kind_n = _turn1_v37_norm(kind)

    if not _turn1_v37_is_trainer(card):
        return False

    name_n = _turn1_v37_card_name_norm(card)
    subtypes = _turn1_v37_card_subtypes_norm(card)

    return kind_n in subtypes or kind_n in name_n


def _turn1_v37_target_phrase_from_filter_blob(blob):
    """
    Extract the searched/selected target phrase.

    Important:
    For Shivery Chill, the full text includes:
      "if this Pokemon is in the Active Spot..."
    We must NOT treat that as a Pokemon search. The target phrase is:
      "up to 2 Basic Water Energy cards"
    """
    b = _turn1_v37_norm(blob)

    patterns = [
        r"search (?:your|the) deck(?: and (?:your )?discard pile)? for (.*?)(?:, reveal| reveal| and reveal|,? and put|,? put| then shuffle| shuffle|\.|$)",
        r"look at the top \d+ cards? of your deck.*?reveal (.*?)(?: card| cards|,| and put| put|$)",
        r"choose (.*?)(?: from (?:your )?deck| from among them|,| and put| put|$)",
    ]

    for pat in patterns:
        m = re.search(pat, b)

        if not m:
            continue

        phrase = m.group(1)
        phrase = re.sub(r"^(up to|exactly)?\s*\d+\s+", "", phrase)
        phrase = re.sub(r"^(a|an|any|one|two)\s+", "", phrase)
        phrase = phrase.strip()

        if phrase:
            return phrase

    return b


def _turn1_v37_raw_text_filter_decision(filt, card):
    """
    Return:
      True  -> raw text explicitly allows this card
      False -> raw text explicitly excludes this card
      None  -> not enough raw-text info; use structured fallback
    """
    if not isinstance(filt, dict):
        return None

    blob = filter_text_blob(filt)

    if not blob:
        return None

    has_raw_text = any(k in filt for k in ["raw_text", "source_text", "text"])
    has_search_language = any(
        phrase in _turn1_v37_norm(blob)
        for phrase in [
            "search your deck",
            "look at the top",
            "reveal",
            "choose",
            "put them into your hand",
            "put it into your hand",
            "put them onto your bench",
            "put it onto your bench",
        ]
    )

    if not has_raw_text and not has_search_language:
        return None

    target = _turn1_v37_target_phrase_from_filter_blob(blob)
    t = _turn1_v37_norm(target)

    tests = []

    # -----------------------------
    # Energy filters
    # -----------------------------
    for typ in _TURN1_V37_TYPE_SYNONYMS:
        if _turn1_v37_has_typed_phrase(t, typ, "energy"):
            if "basic" in t:
                tests.append(lambda c, typ=typ: _turn1_v37_is_basic_energy(c) and _turn1_v37_card_has_type(c, typ))
            else:
                tests.append(lambda c, typ=typ: _turn1_v37_is_energy(c) and _turn1_v37_card_has_type(c, typ))

    if "basic energy" in t:
        tests.append(lambda c: _turn1_v37_is_basic_energy(c))
    elif "energy" in t and not tests:
        tests.append(lambda c: _turn1_v37_is_energy(c))

    # -----------------------------
    # Pokemon filters
    # -----------------------------
    for typ in _TURN1_V37_TYPE_SYNONYMS:
        if _turn1_v37_has_typed_phrase(t, typ, "pokemon"):
            if "basic" in t:
                tests.append(
                    lambda c, typ=typ, target=t: (
                        _turn1_v37_is_basic_pokemon(c)
                        and _turn1_v37_card_has_type(c, typ)
                        and _turn1_v37_hp_ok(c, target)
                    )
                )
            else:
                tests.append(
                    lambda c, typ=typ, target=t: (
                        _turn1_v37_is_pokemon(c)
                        and _turn1_v37_card_has_type(c, typ)
                        and _turn1_v37_hp_ok(c, target)
                    )
                )

    if "pokemon ex" in t:
        tests.append(lambda c: _turn1_v37_is_pokemon_ex(c))

    if "basic pokemon" in t:
        tests.append(lambda c, target=t: _turn1_v37_is_basic_pokemon(c) and _turn1_v37_hp_ok(c, target))
    elif "pokemon" in t and not any(word in t for word in ["energy", "supporter", "item", "stadium"]):
        tests.append(lambda c, target=t: _turn1_v37_is_pokemon(c) and _turn1_v37_hp_ok(c, target))

    # -----------------------------
    # Trainer filters
    # -----------------------------
    if "supporter" in t:
        tests.append(lambda c: _turn1_v37_trainer_kind(c, "supporter"))

    if "item" in t:
        tests.append(lambda c: _turn1_v37_trainer_kind(c, "item"))

    if "stadium" in t:
        tests.append(lambda c: _turn1_v37_trainer_kind(c, "stadium"))

    if "tool" in t:
        tests.append(lambda c: _turn1_v37_trainer_kind(c, "tool") or _turn1_v37_trainer_kind(c, "pokemon tool"))

    if "trainer" in t and not any(word in t for word in ["supporter", "item", "stadium", "tool"]):
        tests.append(lambda c: _turn1_v37_is_trainer(c))

    # -----------------------------
    # Generic "any card" filters
    # -----------------------------
    if not tests and ("any card" in t or t in {"card", "a card"}):
        return True

    if tests:
        return any(test(card) for test in tests)

    # Conservative fallback:
    # If this was clearly a search/select raw-text filter but we could not
    # classify its target, do not let it find arbitrary cards.
    if has_search_language:
        return False

    return None


def filter_allows_card(filt: Dict[str, Any], card: Dict[str, Any]) -> bool:
    """Strict filter matcher for compiled search/select filters.

    v0.37: raw search text is now interpreted as an actual target restriction,
    not as a vague permission to find anything.
    """
    if not filt:
        return True

    raw_decision = _turn1_v37_raw_text_filter_decision(filt, card)

    if raw_decision is not None:
        return bool(raw_decision)

    blob = filter_text_blob(filt)

    # Fighting Gong: "Search your deck for a Basic Fighting Energy card or a
    # Basic Fighting Pokemon...". Kept as explicit fallback.
    if ("basic fighting energy" in blob or "basic f energy" in blob or "basic {f} energy" in blob) and (
        "basic fighting pokemon" in blob or "basic f pokemon" in blob or "basic {f} pokemon" in blob
    ):
        return is_basic_fighting_energy(card) or is_basic_fighting_pokemon(card)

    if "basic fighting energy" in blob or "basic f energy" in blob or "basic {f} energy" in blob:
        return is_basic_fighting_energy(card)

    if "basic fighting pokemon" in blob or "basic f pokemon" in blob or "basic {f} pokemon" in blob:
        return is_basic_fighting_pokemon(card)

    if filt.get("any_card") is True or filt.get("any") is True:
        return True

    st = filt.get("supertype") or filt.get("card_supertype")
    if st and _turn1_v37_norm(st) != _turn1_v37_card_supertype_norm(card):
        return False

    sub = filt.get("subtype") or filt.get("subtypes")
    if sub:
        needed = {_turn1_v37_norm(x) for x in (sub if isinstance(sub, list) else [sub])}
        if not (needed & _turn1_v37_card_subtypes_norm(card)):
            return False

    types = filt.get("types") or filt.get("type")
    if types:
        needed = {_turn1_v37_norm(x) for x in (types if isinstance(types, list) else [types])}
        if not (needed & _turn1_v37_card_types_norm(card)):
            return False

    name_contains = filt.get("name_contains") or filt.get("name")
    if name_contains and _turn1_v37_norm(name_contains) not in _turn1_v37_card_name_norm(card):
        return False

    hp_max = filt.get("hp_max") or filt.get("max_hp")
    if hp_max is not None:
        hp = _turn1_v37_card_hp(card)
        try:
            if hp is None or hp > int(hp_max):
                return False
        except Exception:
            return False

    if filt.get("basic_pokemon") and not _turn1_v37_is_basic_pokemon(card):
        return False

    if filt.get("energy") and not _turn1_v37_is_energy(card):
        return False

    return True

def search_amount(step: Dict[str, Any]) -> int:

    # TURN1_SEARCH_AMOUNT_FROM_TEXT_CANONICAL_V1
    # Canonical amount parser for compiled search effects like Buddy-Buddy Poffin:
    #   amount: {"mode": "from_text"}
    #   source_text: "Search your deck for up to 2 Basic Pokémon with 70 HP or less..."
    #
    # Without this, execute_steps(search_deck) defaults Poffin to selecting only
    # one card even though the printed effect allows up to two.
    try:
        import re
        blob = " ".join(
            str((step or {}).get(k) or "")
            for k in ("source_text", "text", "raw_text", "effect_text", "description")
        ).lower()

        amount_obj = (step or {}).get("amount")
        amount_is_from_text = (
            isinstance(amount_obj, dict)
            and str(amount_obj.get("mode") or "").lower() == "from_text"
        )

        if amount_is_from_text or "up to" in blob:
            m = re.search(r"up to\s+(\d+)", blob)
            if m:
                return max(1, int(m.group(1)))

            words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
            }
            m = re.search(r"up to\s+(one|two|three|four|five)\b", blob)
            if m:
                return words[m.group(1)]
    except Exception:
        pass

    for key in ("amount", "count", "max_cards", "number"):
        if key in step:
            return max(1, amount_value(step[key], default=1))
    text = step_text(step)
    m = re.search(r"(?:search|choose|find).{0,40}?(\d+)", text, flags=re.I)
    return int(m.group(1)) if m else 1


def draw_amount_from_step(step: Dict[str, Any], counts: Optional[Dict[str, int]] = None, coin_heads: int = 0) -> int:
    op = step.get("op")
    if op == "draw_until_hand_size":
        return max(0, amount_value(step.get("target_hand_size"), default=0) - 0)  # scored approximately elsewhere
    for key in ("amount", "cards", "count"):
        if key in step:
            return max(0, amount_value(step[key], default=0, counts=counts, coin_heads=coin_heads))
    text = step_text(step)
    m = re.search(r"draw\s+(\d+)", text, flags=re.I)
    return int(m.group(1)) if m else 0




# ---------------------------------------------------------------------
# TURN1_V67_BENCH_HP_SEARCH_FILTERS
# ---------------------------------------------------------------------
# Broad source-text legality guard used by direct-search scoring. This prevents
# vague compiled filters from treating Bench/HP-limited Pokemon searches as
# Energy access.

def _turn1_v67_norm_text(value):
    try:
        return norm(value)
    except Exception:
        return str(value or "").lower().replace("pokémon", "pokemon").strip()


def _turn1_v67_flatten_strings(value, depth=0):
    if value is None or depth > 4:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        preferred = [
            "name", "card_name", "text", "raw_text", "source_text", "combined_text",
            "rules", "abilities_text", "attacks_text", "effect_text", "description",
            "filter", "card_filter", "selection", "target", "targets",
        ]
        for key in preferred:
            if key in value:
                parts.append(_turn1_v67_flatten_strings(value.get(key), depth + 1))
        for key, val in value.items():
            if key not in preferred:
                parts.append(_turn1_v67_flatten_strings(val, depth + 1))
        return " ".join(p for p in parts if p)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_turn1_v67_flatten_strings(v, depth + 1) for v in value)
    return str(value)


def _turn1_v67_card_source_blob(card, filt=None, step=None):
    parts = []
    if isinstance(step, dict):
        for key in ("text", "raw_text", "source_text", "effect_text", "description"):
            val = step.get(key)
            if val:
                parts.append(str(val))
        try:
            parts.append(step_text(step))
        except Exception:
            pass
        parts.append(_turn1_v67_flatten_strings(step))

    try:
        parts.append(filter_text_blob(filt))
    except Exception:
        parts.append(_turn1_v67_flatten_strings(filt))

    if isinstance(card, dict):
        parts.append(_turn1_v67_flatten_strings(card))
        try:
            for eff in iter_effects(card):
                try:
                    parts.append(ability_text_blob(eff))
                except Exception:
                    pass
                parts.append(_turn1_v67_flatten_strings(eff))
        except Exception:
            pass

    return _turn1_v67_norm_text(" ".join(p for p in parts if p))


def _turn1_v67_target_phrase_from_search_text(blob):
    b = _turn1_v67_norm_text(blob)
    patterns = [
        r"search (?:your|the) deck(?: and (?:your )?discard pile)? for (.*?)(?:, reveal| reveal| and reveal|,? and put|,? put| then shuffle| shuffle|\.|$)",
        r"look at the top \d+ cards? of your deck.*?reveal (.*?)(?: card| cards|,| and put| put|$)",
        r"choose (.*?)(?: from (?:your )?deck| from among them|,| and put| put|$)",
    ]
    for pat in patterns:
        m = re.search(pat, b)
        if not m:
            continue
        phrase = m.group(1)
        phrase = re.sub(r"^(up to|exactly)?\s*\d+\s+", "", phrase).strip()
        phrase = re.sub(r"^(a|an|any|one|two)\s+", "", phrase).strip()
        if phrase:
            return phrase
    return b


def _turn1_v67_card_hp(card):
    vals = []
    if isinstance(card, dict):
        vals.extend([card.get("hp"), card.get("HP"), card.get("raw_hp")])
        for key in ("identity", "gameplay", "raw_card", "source"):
            obj = card.get(key)
            if isinstance(obj, dict):
                vals.extend([obj.get("hp"), obj.get("HP"), obj.get("raw_hp")])
    for val in vals:
        if val is None:
            continue
        m = re.search(r"\d+", str(val))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    return None


def _turn1_v67_is_pokemon(card):
    try:
        return _turn1_v67_norm_text(card_supertype(card)) in {"pokemon", "pokémon"}
    except Exception:
        return False


def _turn1_v67_is_basic_pokemon(card):
    try:
        return bool(is_basic_pokemon(card))
    except Exception:
        if not _turn1_v67_is_pokemon(card):
            return False
        try:
            subs = {_turn1_v67_norm_text(x) for x in card_subtypes(card)}
        except Exception:
            subs = set()
        try:
            name_n = _turn1_v67_norm_text(card_name(card))
        except Exception:
            name_n = ""
        return "basic" in subs or name_n.startswith("basic ")


def _turn1_v67_source_text_allows_target(action_card, filt, target_card, step=None):
    blob = _turn1_v67_card_source_blob(action_card, filt=filt, step=step)
    if not blob:
        return True

    target = _turn1_v67_target_phrase_from_search_text(blob)
    target_n = _turn1_v67_norm_text(target)

    if "onto your bench" in blob or "to your bench" in blob:
        if not _turn1_v67_is_pokemon(target_card):
            return False

    hp_match = re.search(r"(\d+)\s*hp\s*or\s*less", target_n) or re.search(r"(\d+)\s*hp\s*or\s*less", blob)
    if hp_match:
        if not _turn1_v67_is_pokemon(target_card):
            return False
        hp = _turn1_v67_card_hp(target_card)
        if hp is None or hp > int(hp_match.group(1)):
            return False

    if "basic pokemon" in target_n or "basic pokémon" in target_n:
        if not _turn1_v67_is_basic_pokemon(target_card):
            return False

    return True


def card_directly_searches_target(card: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    # TURN1_V67_BENCH_HP_SEARCH_FILTERS
    target_cards = [c for c in deck if target_matches(c, target_norm)]
    if not target_cards:
        return False
    if card_specific_directly_searches_target(card, target_norm, deck):
        return True
    for step in meaningful_steps(card):
        if step.get("op") in {"search_deck", "choose_cards", "put_card_into_hand", "put_card_onto_bench"}:
            filt = extract_filter(step)
            if any(
                filter_allows_card(filt, tc)
                and _turn1_v67_source_text_allows_target(card, filt, tc, step=step)
                for tc in target_cards
            ):
                return True
    return False


def card_draw_power(card: Dict[str, Any]) -> int:
    total = 0
    for step in flatten_steps(list(iter_effects(card))):
        op = step.get("op")
        if op in {"draw_cards", "draw_cards_per_coin_heads"}:
            total += draw_amount_from_step(step, coin_heads=1)
        elif op == "draw_until_hand_size":
            total += amount_value(step.get("target_hand_size"), default=0)
        elif op == "draw_until_hand_size_matches":
            total += 3  # unknown opponent hand; small heuristic
    return total


def card_has_search(card: Dict[str, Any]) -> bool:
    return any(s.get("op") == "search_deck" for s in flatten_steps(list(iter_effects(card))))


def card_known_discard_cost(card: Dict[str, Any]) -> int:
    """Return a conservative required discard count for common search cards.

    The compiler usually emits this as a discard_cards step with
    required_to_play=True, but this helper lets the turn-1 policy reason
    about playability before choosing the card.
    """
    name_n = norm(card_name(card))
    if name_n == "ultra ball":
        return 2

    cost = 0
    for step in flatten_steps(list(iter_effects(card))):
        if step.get("op") not in {"discard_cards", "discard_card"}:
            continue
        required = bool(step.get("required_to_play")) or "only if you discard" in norm(step_text(step))
        if not required:
            continue
        selection = step.get("selection") if isinstance(step.get("selection"), dict) else {}
        amount = step.get("amount") or selection.get("value") or step.get("count")
        cost = max(cost, amount_value(amount, default=1))
    return cost


def has_enough_discard_fodder(hand: Sequence[Dict[str, Any]], card: Dict[str, Any], target_norm: str) -> bool:
    cost = card_known_discard_cost(card)
    if cost <= 0:
        return True
    others = [c for c in hand if c is not card and not target_matches(c, target_norm)]
    return len(others) >= cost


def target_is_pokemon_in_pool(target_norm: str, pool: Sequence[Dict[str, Any]]) -> bool:
    return any(target_matches(c, target_norm) and card_supertype(c) == "Pokémon" for c in pool)


def card_specific_directly_searches_target(card: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    """Fallback for common cards whose compiled filters may be too vague.

    Directly searches target means: this card can put the target into hand.
    Ciphermaniac is intentionally NOT direct because it only puts cards on top
    of the deck; it needs an immediate draw effect such as Run Errand.
    """
    if is_ultra_ball(card):
        return target_is_pokemon_in_pool(target_norm, deck)
    if is_cyrano(card):
        return target_is_pokemon_ex_in_pool(target_norm, deck)
    return False


def card_has_specific_play_effect(card: Dict[str, Any]) -> bool:
    """True for common cards/abilities we model explicitly for the target-finding scenario."""
    return any([
        is_ultra_ball(card),
        is_cyrano(card),
        is_ciphermaniac(card),
        is_lillies_determination(card),
        is_crispin(card),
        is_meowth_ex(card),
    ])


def card_can_be_played_from_hand(card: Dict[str, Any], going: str, supporter_used: bool) -> bool:
    # Meowth ex is a Basic Pokémon that can be played to the Bench to trigger Last-Ditch Catch.
    if is_meowth_ex(card):
        return True

    # Main target-finder v0.6 focuses on Trainer cards from hand plus explicit Pokémon abilities.
    if not is_trainer(card):
        return False
    if not (card_has_play_effect(card) or card_has_specific_play_effect(card)):
        return False
    if is_supporter(card):
        if supporter_used:
            return False
        # Current Pokémon rules: player going first cannot play Supporter on their first turn.
        if going == "first":
            return False
    return True


# -----------------------------
# Scenario state and executor
# -----------------------------


@dataclass
class SimState:
    deck: List[Dict[str, Any]]
    hand: List[Dict[str, Any]]
    prizes: List[Dict[str, Any]]
    discard: List[Dict[str, Any]] = field(default_factory=list)
    supporter_used: bool = False
    found: bool = False
    found_stage: Optional[str] = None
    line: List[str] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    memory_cards: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    coin_heads: int = 0
    actions_used: int = 0
    active: Optional[Dict[str, Any]] = None
    bench: List[Dict[str, Any]] = field(default_factory=list)
    run_errand_used: bool = False
    last_ditch_used: bool = False
    abilities_used: set = field(default_factory=set)

    def has_target_in_hand(self, target_norm: str) -> bool:
        return any(target_matches(c, target_norm) for c in self.hand)

    def target_in_deck(self, target_norm: str) -> bool:
        return any(target_matches(c, target_norm) for c in self.deck)


def remove_first_matching(cards: List[Dict[str, Any]], predicate) -> Optional[Dict[str, Any]]:
    for i, c in enumerate(cards):
        if predicate(c):
            return cards.pop(i)
    return None


def choose_optimal_active(opening: Sequence[Dict[str, Any]], target_norm: str) -> Optional[Dict[str, Any]]:
    """Choose an Active Pokémon for the target-finding objective.

    If the target is not already found, Mega Kangaskhan ex is the best Active
    because Run Errand draws 2. If Meowth ex is in the opener, prefer another
    Basic as Active so Meowth can be benched during the turn and trigger
    Last-Ditch Catch.
    """
    basics = [c for c in opening if is_basic_pokemon(c)]
    if not basics:
        return None
    # Preserve target-finding cards in hand when possible.
    non_target_basics = [c for c in basics if not target_matches(c, target_norm)]
    pool = non_target_basics or basics
    for c in pool:
        if is_mega_kangaskhan_ex(c):
            return c
    non_meowth = [c for c in pool if not is_meowth_ex(c)]
    if non_meowth:
        return non_meowth[0]
    return pool[0]


def can_use_run_errand(st: SimState) -> bool:
    return bool(st.active is not None and is_mega_kangaskhan_ex(st.active) and not st.run_errand_used)


def use_run_errand(st: SimState, target_norm: str, stage: str) -> None:
    """Mega Kangaskhan ex — Run Errand: if Active, draw 2 once per turn."""
    if not can_use_run_errand(st):
        return
    st.run_errand_used = True
    st.actions_used += 1
    st.line.append("Run Errand")
    draw_cards(st, 2, stage)
    st.log.append({"event": "use_ability", "ability": "Run Errand", "stage": stage, "source": "Mega Kangaskhan ex"})
    if st.has_target_in_hand(target_norm):
        st.found = True
        st.found_stage = stage


def run_errand_score(st: SimState, target_norm: str) -> float:
    if not can_use_run_errand(st):
        return -1.0
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0 or not st.deck:
        return -1.0
    # If target is actually in the top two after known ordering, use it now.
    if any(target_matches(c, target_norm) for c in st.deck[:2]):
        return 9000.0
    # Otherwise it is a normal two-card dig line. Keep it below direct search
    # and below Lillie/Cipher lines, but above doing nothing.
    n = min(2, len(st.deck))
    return 1800.0 * (1.0 - hypergeom_zero(len(st.deck), target_remaining, n))



# -----------------------------
# Generic Pokémon Ability layer
# -----------------------------


def _source_ability_rows(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return raw/source ability rows from compiled card metadata.

    Some cards have printed abilities in `sources.abilities` / raw API data but
    no compiled ability in `compiled_effects`. For a target-finding simulator,
    it is better to synthesize conservative draw/search/look effects from the
    printed ability text than to ignore the ability entirely.
    """
    rows: List[Dict[str, Any]] = []

    sources = card.get("sources") if isinstance(card.get("sources"), dict) else {}
    for row in sources.get("abilities") or []:
        if isinstance(row, dict):
            rows.append(row)

    raw = sources.get("raw_api_card") if isinstance(sources.get("raw_api_card"), dict) else {}
    for row in raw.get("abilities") or []:
        if isinstance(row, dict):
            rows.append(row)

    # Older/generated objects sometimes keep raw ability rows outside `sources`.
    for key in ("abilities", "raw_abilities", "raw_abilities_json"):
        val = card.get(key)
        if isinstance(val, list):
            for row in val:
                if isinstance(row, dict):
                    rows.append(row)
        elif isinstance(val, str) and val.strip().startswith("["):
            try:
                parsed = json.loads(val)
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict):
                        rows.append(row)

    # Deduplicate by name + text.
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (norm(row.get("name")), norm(row.get("text")))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(row)
    return out


def _steps_from_printed_ability_text(text: str) -> List[Dict[str, Any]]:
    """Conservatively synthesize target-finding steps from printed ability text.

    This intentionally covers only effects that matter for finding cards by turn 1:
    drawing, searching deck into hand, looking at top cards, and ordering cards.
    Costs and board requirements stay in the effect text so the generic ability
    requirement/cost parser can still enforce them.
    """
    blob = norm(text)
    steps: List[Dict[str, Any]] = []

    # Draw N cards. Example: Lunar Cycle -> Draw 3 cards.
    for m in re.finditer(r"draw (\d+) cards?", blob):
        steps.append({"op": "draw_cards", "amount": int(m.group(1)), "source_text": text})

    # Draw until you have N cards in hand.
    m = re.search(r"draw cards until you have (\d+) cards? in your hand", blob)
    if m:
        steps.append({"op": "draw_until_hand_size", "target_hand_size": int(m.group(1)), "source_text": text})

    # Search your deck for ... put it/them into your hand. The filter is text-only,
    # but filter_allows_card already understands useful raw_text filters such as
    # Pokémon, Pokémon ex, Basic Energy, etc.
    if "search your deck" in blob and ("put" in blob and "hand" in blob):
        raw_filter = text
        amount = 1
        m = re.search(r"up to (\d+)", blob) or re.search(r"for (\d+)", blob)
        if m:
            amount = int(m.group(1))
        steps.append({"op": "search_deck", "amount": amount, "filter": {"raw_text": raw_filter}, "source_text": text})

    # Look at the top N cards.
    m = re.search(r"look at the top (\d+) cards?", blob)
    if m:
        steps.append({"op": "look_at_top_cards", "amount": int(m.group(1)), "source_text": text})

    # If the text explicitly lets you reorder/top-deck cards, expose that as reorder.
    if "put" in blob and "top of your deck" in blob:
        steps.append({"op": "reorder_cards", "source_text": text})

    return steps


def _source_ability_effects(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    effects: List[Dict[str, Any]] = []
    for idx, row in enumerate(_source_ability_rows(card)):
        name = str(row.get("name") or f"Ability {idx + 1}")
        text = str(row.get("text") or "")
        steps = _steps_from_printed_ability_text(text)
        if not steps:
            continue
        effects.append({
            "effect_id": f"{card_id(card)}::source_ability::{idx}",
            "effect_kind": "ability_activated",
            "kind": "ability_activated",
            "name": name,
            "ability_name": name,
            "text": text,
            "source_text": text,
            "steps": steps,
            "_synthetic_from_source_ability": True,
        })
    return effects


def ability_dedupe_key(effect: Dict[str, Any]) -> Tuple[str, str]:
    """Stable identity for duplicate compiled/source ability rows.

    Compiler v0.9 can now emit abilities such as Lunar Cycle directly, while
    v0.14 also adds a raw-text fallback from sources.abilities. If both exist,
    they describe the same printed ability and must not both become playable
    actions. Prefer the normalized printed text when available; otherwise fall
    back to the normalized ability name.
    """
    text = norm(ability_text_blob(effect))
    if text:
        return ("text", text)
    name = norm(ability_name_from_effect(effect))
    if name:
        return ("name", name)
    return ("id", str(effect.get("effect_id") or effect.get("id") or id(effect)))


def ability_effects(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ability effects, using compiled effects plus source-text fallbacks.

    v0.15 fix: if the compiler already emitted a useful ability, do not also add
    the source-text fallback for the same printed text. This prevents duplicate
    lines such as `Lunar Cycle -> Once during your turn...` and ensures usage
    limits apply to a single stable ability identity.
    """
    out: List[Dict[str, Any]] = []
    if card_supertype(card) != "Pokémon":
        return out

    seen_keys = set()
    for eff in iter_effects(card):
        kind = str(eff.get("effect_kind") or eff.get("kind") or "").lower()
        eid = str(eff.get("effect_id") or eff.get("id") or "").lower()
        if "ability" in kind or "::ability" in eid:
            if not effect_is_trivial_rule(eff):
                key = ability_dedupe_key(eff)
                if key not in seen_keys:
                    out.append(eff)
                    seen_keys.add(key)

    for eff in _source_ability_effects(card):
        key = ability_dedupe_key(eff)
        if key in seen_keys:
            continue
        out.append(eff)
        seen_keys.add(key)

    return out


def ability_name_from_effect(effect: Dict[str, Any]) -> str:
    for key in ("name", "ability_name", "label", "title"):
        if effect.get(key):
            return str(effect.get(key))

    # Compiler effects usually store the printed ability name under source.name.
    # Use that before falling back to raw text or effect_id so lines show
    # `Lunar Cycle`, not `Once during your turn, if you have Solro...`.
    source = effect.get("source") if isinstance(effect.get("source"), dict) else {}
    for key in ("name", "ability_name", "label", "title"):
        if source.get(key):
            return str(source.get(key))

    # Some older rows keep source metadata under other nested keys.
    for nested_key in ("printed", "metadata"):
        nested = effect.get(nested_key) if isinstance(effect.get(nested_key), dict) else {}
        for key in ("name", "ability_name", "label", "title"):
            if nested.get(key):
                return str(nested.get(key))

    # Many compiled effects do not carry a clean ability_name field but still
    # keep the printed text. Prefer a printed name like "Lunar Cycle [Ability]"
    # over a generic effect id such as "ability::1".
    raw_text_parts = [str(effect.get("text") or ""), str(effect.get("source_text") or "")]
    for step in flatten_steps(effect):
        raw_text_parts.extend([str(step.get("text") or ""), str(step.get("source_text") or "")])
    raw_text = " ".join(p for p in raw_text_parts if p).strip()
    if raw_text:
        m = re.search(r"([A-Z][A-Za-z0-9'’\- ]{1,48})\s*\[Ability\]", raw_text)
        if m:
            return m.group(1).strip()
        m = re.search(r"^([A-Z][A-Za-z0-9'’\- ]{1,48})\s*:", raw_text)
        if m:
            return m.group(1).strip()

    eid = str(effect.get("effect_id") or effect.get("id") or "")
    if "::" in eid:
        parts = [p for p in eid.split("::") if p]
        if parts:
            candidate = parts[-1].replace("_", " ").title()
            if not candidate.isdigit():
                return candidate
    if raw_text:
        return raw_text[:40]
    return "Ability"


def is_teal_mask_ogerpon_ex(card: Dict[str, Any]) -> bool:
    return is_named(card, "Teal Mask Ogerpon ex")


def is_basic_grass_energy(card: Dict[str, Any]) -> bool:
    if card_supertype(card) != "Energy":
        return False
    name_n = norm(card_name(card))
    types = {norm(x) for x in card_types(card)}
    return "grass" in name_n or "grass" in types


ENERGY_TYPE_NAMES = ["Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal"]


def is_basic_typed_energy(card: Dict[str, Any], energy_type: Optional[str] = None) -> bool:
    """Return True for Basic Energy, optionally restricted to a type.

    This covers real API cards and project-local proxy Basic Energy cards.
    """
    if card_supertype(card) != "Energy":
        return False
    subtypes_n = {norm(x) for x in card_subtypes(card)}
    name_n = norm(card_name(card))
    if "basic" not in subtypes_n and "basic" not in name_n:
        return False
    if energy_type is None:
        return True
    et = norm(energy_type)
    types_n = {norm(x) for x in card_types(card)}
    return et in name_n or et in types_n


def ability_text_blob(effect: Dict[str, Any]) -> str:
    parts = [
        str(effect.get("name") or ""),
        str(effect.get("ability_name") or ""),
        str(effect.get("text") or ""),
        str(effect.get("source_text") or ""),
    ]
    for step in flatten_steps(effect):
        parts.extend([
            str(step.get("text") or ""),
            str(step.get("source_text") or ""),
            step_text(step),
        ])
    return " ".join(p for p in parts if p).strip()


def ability_norm_blob(effect: Dict[str, Any]) -> str:
    return norm(ability_text_blob(effect))


def cards_known_to_state(st: SimState) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    if st.active is not None:
        cards.append(st.active)
    cards.extend(list(st.bench))
    cards.extend(list(st.hand))
    cards.extend(list(st.deck))
    cards.extend(list(st.discard))
    cards.extend(list(st.prizes))
    return cards


def unique_pokemon_names_known_to_state(st: SimState) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in cards_known_to_state(st):
        if card_supertype(c) != "Pokémon":
            continue
        nm = card_name(c)
        key = norm(nm)
        if key and key not in seen:
            seen.add(key)
            out.append(nm)
    # Longest first prevents "Meowth" from matching before "Meowth ex".
    out.sort(key=lambda x: len(norm(x)), reverse=True)
    return out


def pokemon_in_play_by_name(st: SimState, name: str) -> bool:
    target = norm(name)
    return any(norm(card_name(c)) == target for c in in_play_pokemon(st))


def bench_basic_by_name_from_hand(st: SimState, name: str, stage: str) -> bool:
    target = norm(name)
    if len(st.bench) >= bench_capacity(st):
        return False
    for c in list(st.hand):
        if norm(card_name(c)) == target and is_basic_pokemon(c):
            st.hand.remove(c)
            st.bench.append(c)
            st.actions_used += 1
            st.line.append(card_name(c))
            st.log.append({"event": "ability_requirement_benched", "card": card_name(c), "stage": stage})
            return True
    return False


def ability_required_pokemon_names(effect: Dict[str, Any], st: SimState, source: Optional[Dict[str, Any]] = None) -> List[str]:
    """Infer printed "if you have X in play" requirements.

    This is generic rather than Lunar Cycle-specific. It scans the printed / step
    text for known Pokémon names near "in play" / "on your Bench" language.
    """
    blob = ability_norm_blob(effect)
    if not any(token in blob for token in ["in play", "on your bench", "on the bench", "on your active"]):
        return []
    source_key = norm(card_name(source)) if source else ""
    reqs: List[str] = []
    seen = set()
    for name in unique_pokemon_names_known_to_state(st):
        key = norm(name)
        if not key or key == source_key or key in seen:
            continue
        if key in blob:
            reqs.append(name)
            seen.add(key)
    return reqs


def ability_requirements_can_be_met(effect: Dict[str, Any], st: SimState, source: Dict[str, Any]) -> bool:
    for name in ability_required_pokemon_names(effect, st, source):
        if pokemon_in_play_by_name(st, name):
            continue
        # We only auto-set up Basic Pokémon from hand. Fetching requirements from
        # deck is handled through normal search/enabler actions; evolutions are not
        # free and should not be assumed turn 1.
        if len(st.bench) >= bench_capacity(st):
            return False
        if not any(norm(card_name(c)) == norm(name) and is_basic_pokemon(c) for c in st.hand):
            return False
    return True


def prepare_ability_requirements(effect: Dict[str, Any], st: SimState, source: Dict[str, Any], stage: str) -> bool:
    for name in ability_required_pokemon_names(effect, st, source):
        if pokemon_in_play_by_name(st, name):
            st.log.append({"event": "ability_requirement_already_in_play", "card": name, "stage": stage})
            continue
        if not bench_basic_by_name_from_hand(st, name, stage):
            st.log.append({"event": "ability_requirement_missing", "card": name, "stage": stage})
            return False
    return True


def ability_discard_energy_costs(effect: Dict[str, Any]) -> List[Tuple[Optional[str], int]]:
    """Infer costs like "discard a Basic Fighting Energy card from your hand".

    Returns (energy_type, amount). energy_type None means any Basic Energy.
    """
    blob_raw = ability_text_blob(effect)
    blob = norm(blob_raw)
    if "discard" not in blob or "energy" not in blob or "hand" not in blob:
        return []
    costs: List[Tuple[Optional[str], int]] = []
    for typ in ENERGY_TYPE_NAMES:
        typ_n = norm(typ)
        if re.search(rf"discard (?:a |an |1 )?(?:basic )?{typ_n} energy", blob):
            m = re.search(rf"discard (\d+) (?:basic )?{typ_n} energy", blob)
            costs.append((typ, int(m.group(1)) if m else 1))
    if not costs and re.search(r"discard (?:a |an |1 )?basic energy", blob):
        m = re.search(r"discard (\d+) basic energy", blob)
        costs.append((None, int(m.group(1)) if m else 1))
    return costs


def ability_generic_discard_card_cost(effect: Dict[str, Any]) -> int:
    blob = ability_norm_blob(effect)
    if "discard" not in blob or "hand" not in blob:
        return 0
    # Do not double-count energy-specific costs.
    if ability_discard_energy_costs(effect):
        return 0
    m = re.search(r"discard (\d+) cards? from your hand", blob)
    if m:
        return int(m.group(1))
    if re.search(r"discard (?:a|1) cards? from your hand", blob):
        return 1
    return 0


def ability_costs_can_be_paid(effect: Dict[str, Any], st: SimState, target_norm: str) -> bool:
    for typ, amount in ability_discard_energy_costs(effect):
        pool = [c for c in st.hand if not target_matches(c, target_norm) and is_basic_typed_energy(c, typ)]
        if len(pool) < amount:
            return False
    generic_cost = ability_generic_discard_card_cost(effect)
    if generic_cost:
        pool = [c for c in st.hand if not target_matches(c, target_norm)]
        if len(pool) < generic_cost:
            return False
    return True


def pay_ability_costs(effect: Dict[str, Any], st: SimState, target_norm: str, stage: str) -> bool:
    paid_any = False
    for typ, amount in ability_discard_energy_costs(effect):
        discarded: List[str] = []
        for _ in range(amount):
            candidates = [c for c in st.hand if not target_matches(c, target_norm) and is_basic_typed_energy(c, typ)]
            if not candidates:
                return False
            candidates.sort(key=card_name)
            chosen = candidates[0]
            st.hand.remove(chosen)
            st.discard.append(chosen)
            discarded.append(card_name(chosen))
        paid_any = True
        st.log.append({"event": "ability_cost_discard", "stage": stage, "cost": "basic_energy", "energy_type": typ or "Any", "discarded": discarded})
    generic_cost = ability_generic_discard_card_cost(effect)
    if generic_cost:
        discarded = []
        for _ in range(generic_cost):
            candidates = [c for c in st.hand if not target_matches(c, target_norm)]
            if not candidates:
                return False
            candidates.sort(key=lambda c: (card_has_play_effect(c), card_name(c)))
            chosen = candidates[0]
            st.hand.remove(chosen)
            st.discard.append(chosen)
            discarded.append(card_name(chosen))
        paid_any = True
        st.log.append({"event": "ability_cost_discard", "stage": stage, "cost": "card", "discarded": discarded})
    if paid_any:
        # If the compiled effect also contains a discard_cards/discard_card cost
        # step, skip the next such step to avoid paying the same printed cost twice.
        st.counts["_skip_next_discard_cost"] = int(st.counts.get("_skip_next_discard_cost", 0) or 0) + 1
    return True


def ability_usage_key(source: Dict[str, Any], effect: Dict[str, Any]) -> Tuple[Any, ...]:
    ability_name = ability_name_from_effect(effect)
    blob = ability_norm_blob(effect)
    name_key = norm(ability_name)
    if "cant use more than 1" in blob or "can t use more than 1" in blob or "cant use more than one" in blob or "can t use more than one" in blob:
        return ("ability_global_once", name_key)
    return ("ability_source", id(source), str(effect.get("effect_id") or effect.get("id") or name_key))


def ability_ready_for_target_finding(effect: Dict[str, Any], st: SimState, source: Dict[str, Any], target_norm: str) -> bool:
    return ability_requirements_can_be_met(effect, st, source) and ability_costs_can_be_paid(effect, st, target_norm)


def bench_capacity(st: SimState) -> int:
    # Area Zero's expanded Bench is not modeled yet; normal Bench cap only.
    return 5


def in_play_pokemon(st: SimState) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if st.active is not None:
        out.append(st.active)
    out.extend(list(st.bench))
    return out


def ability_draw_power(effect: Dict[str, Any]) -> int:
    total = 0
    for step in flatten_steps(effect):
        op = step.get("op")
        if op in {"draw_cards", "draw_cards_per_coin_heads"}:
            total += draw_amount_from_step(step, coin_heads=1)
        elif op == "draw_until_hand_size":
            total += amount_value(step.get("target_hand_size"), default=0)
        elif op == "draw_until_hand_size_matches":
            total += 3
    return total


def ability_directly_searches_target(effect: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    target_cards = [c for c in deck if target_matches(c, target_norm)]
    if not target_cards:
        return False
    for step in flatten_steps(effect):
        if step.get("op") in {"search_deck", "choose_cards", "put_card_into_hand"}:
            filt = extract_filter(step)
            if any(filter_allows_card(filt, tc) for tc in target_cards):
                return True
    return False


def ability_has_target_finding_ops(effect: Dict[str, Any]) -> bool:
    for step in flatten_steps(effect):
        if step.get("op") in {
            "draw_cards", "draw_cards_per_coin_heads", "draw_until_hand_size", "draw_until_hand_size_matches",
            "search_deck", "choose_cards", "put_card_into_hand",
            "look_at_top_cards", "look_at_cards", "reorder_cards", "move_card", "move_cards",
        }:
            return True
    return False


def card_has_usable_ability(card: Dict[str, Any]) -> bool:
    if is_mega_kangaskhan_ex(card) or is_meowth_ex(card) or is_teal_mask_ogerpon_ex(card):
        return True
    return any(ability_has_target_finding_ops(eff) for eff in ability_effects(card))


def can_bench_basic_for_ability(st: SimState, card: Dict[str, Any], target_norm: str) -> bool:
    if not is_basic_pokemon(card):
        return False
    if len(st.bench) >= bench_capacity(st):
        return False
    if card not in st.hand:
        return False
    # If this card is the target, the trial would already have succeeded from hand.
    if target_matches(card, target_norm):
        return False
    return card_has_usable_ability(card)


def score_generic_ability(st: SimState, source: Dict[str, Any], effect: Dict[str, Any], target_norm: str) -> float:
    """Heuristic score for any compiled Pokémon Ability.

    v0.12 gates generic abilities by inferred printed requirements/costs before
    scoring them. This prevents over-counting abilities that require another
    Pokémon in play or an Energy/card discard cost, while allowing engines like
    Lunar Cycle to work when the required board + hand state is actually present
    or can be set up from hand.
    """
    if not ability_ready_for_target_finding(effect, st, source, target_norm):
        return -1.0
    if ability_directly_searches_target(effect, target_norm, st.deck):
        return 8800.0
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0
    deck_size = max(1, len(st.deck))
    draw_power = ability_draw_power(effect)
    draw_score = 1000.0 * (1.0 - hypergeom_zero(deck_size, target_remaining, min(draw_power, deck_size))) if draw_power else 0.0
    look_score = 0.0
    for step in flatten_steps(effect):
        if step.get("op") in {"look_at_top_cards", "look_at_cards"}:
            n = amount_value(step.get("amount") or step.get("count") or step.get("number"), default=1)
            if any(target_matches(c, target_norm) for c in st.deck[:max(0, n)]):
                look_score = max(look_score, 100.0)
    return max(draw_score, look_score)

def can_use_teal_dance(st: SimState) -> bool:
    # Once per Teal Mask Ogerpon ex in play. This is good enough for turn-1 target finding.
    used = int(st.counts.get("teal_dance_used", 0) or 0)
    available_sources = sum(1 for c in in_play_pokemon(st) if is_teal_mask_ogerpon_ex(c))
    if used >= available_sources:
        return False
    return any(is_basic_grass_energy(c) for c in st.hand)


def teal_dance_score(st: SimState, target_norm: str) -> float:
    if not can_use_teal_dance(st):
        return -1.0
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0
    return 1000.0 * (1.0 - hypergeom_zero(max(1, len(st.deck)), target_remaining, 1))


def use_teal_dance(st: SimState, target_norm: str, stage: str) -> None:
    if not can_use_teal_dance(st):
        return
    energy = None
    for c in list(st.hand):
        if is_basic_grass_energy(c):
            energy = c
            break
    if energy is None:
        return
    st.hand.remove(energy)
    st.counts["teal_dance_used"] = int(st.counts.get("teal_dance_used", 0) or 0) + 1
    st.actions_used += 1
    st.line.append("Teal Dance")
    st.log.append({"event": "use_ability", "ability": "Teal Dance", "stage": stage, "source": "Teal Mask Ogerpon ex", "attached": card_name(energy)})
    draw_cards(st, 1, stage)
    if st.has_target_in_hand(target_norm):
        st.found = True
        st.found_stage = stage


def generic_ability_candidates(st: SimState, target_norm: str) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for source in in_play_pokemon(st):
        for idx, eff in enumerate(ability_effects(source)):
            key = ability_usage_key(source, eff)
            if key in st.abilities_used:
                continue
            # These have explicit models so we do not double-count them.
            name_n = norm(ability_name_from_effect(eff) + " " + ability_text_blob(eff))
            if "run errand" in name_n or "teal dance" in name_n or "last ditch" in name_n:
                continue
            score = score_generic_ability(st, source, eff, target_norm)
            if score > 0:
                out.append((score, source, eff))
    return out


def use_generic_ability(st: SimState, source: Dict[str, Any], effect: Dict[str, Any], rng: random.Random, target_norm: str, going: str, enable_chain_search: bool) -> None:
    key = ability_usage_key(source, effect)
    if key in st.abilities_used:
        return
    ability_name = ability_name_from_effect(effect)
    stage = f"after_use_{ability_name}"
    if not prepare_ability_requirements(effect, st, source, stage):
        return
    if not pay_ability_costs(effect, st, target_norm, stage):
        return
    st.abilities_used.add(key)
    st.actions_used += 1
    st.line.append(ability_name)
    st.log.append({"event": "use_ability", "ability": ability_name, "stage": stage, "source": card_name(source)})
    execute_steps(st, iter_steps(effect), rng, target_norm, going, stage, enable_chain_search)

def bench_basic_for_ability(st: SimState, card: Dict[str, Any], rng: random.Random, target_norm: str, going: str, enable_chain_search: bool) -> None:
    if not can_bench_basic_for_ability(st, card, target_norm):
        return
    # Preserve special Meowth handling from the target-finder policy.
    if is_meowth_ex(card):
        play_card(st, card, rng, target_norm, going, enable_chain_search)
        return
    st.hand.remove(card)
    st.bench.append(card)
    st.actions_used += 1
    st.line.append(card_name(card))
    stage = f"after_bench_{card_name(card)}"
    st.log.append({"event": "play_basic_to_bench", "card": card_name(card), "stage": stage})
    if is_teal_mask_ogerpon_ex(card) and can_use_teal_dance(st):
        use_teal_dance(st, target_norm, f"{stage}_then_Teal_Dance")
        return
    cands = [x for x in generic_ability_candidates(st, target_norm) if x[1] is card]
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        _, source, eff = cands[0]
        use_generic_ability(st, source, eff, rng, target_norm, going, enable_chain_search)


def bench_basic_ability_score(st: SimState, card: Dict[str, Any], target_norm: str, going: str) -> float:
    if not can_bench_basic_for_ability(st, card, target_norm):
        return -1.0
    if is_meowth_ex(card):
        return score_playable_card(card, st, target_norm, going, True)
    if is_teal_mask_ogerpon_ex(card):
        if any(is_basic_grass_energy(c) for c in st.hand if c is not card):
            return teal_dance_score(st, target_norm) if card in st.bench else 1000.0 * (1.0 - hypergeom_zero(max(1, len(st.deck)), sum(1 for x in st.deck if target_matches(x, target_norm)), 1))
        return -1.0
    best = -1.0
    for eff in ability_effects(card):
        best = max(best, score_generic_ability(st, card, eff, target_norm))
    return best


# -----------------------------
# Ability requirement search chains
# -----------------------------


def search_card_can_find_card(searcher: Dict[str, Any], wanted: Dict[str, Any]) -> bool:
    # TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION
    # Broad replacement for the previous filter-only helper. A searcher can find
    # wanted only if its own effect can legally reach that specific card.
    if not isinstance(searcher, dict) or not isinstance(wanted, dict):
        return False
    return bool(_turn1_v70_source_can_fetch_candidate(searcher, wanted, [wanted]))


def ability_missing_basic_requirement_cards_in_deck(effect: Dict[str, Any], st: SimState, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find missing required Basic Pokémon for an ability that are still in deck.

    Example: Lunar Cycle needs Solrock in play. If Lunatone is in play and
    Solrock is in deck, this returns the Solrock card so a search enabler can
    fetch and bench it before using the ability.
    """
    missing: List[Dict[str, Any]] = []
    seen = set()
    for name in ability_required_pokemon_names(effect, st, source):
        if pokemon_in_play_by_name(st, name):
            continue
        key = norm(name)
        if key in seen:
            continue
        # If it is already in hand, the normal ability preparation path can bench it.
        if any(norm(card_name(c)) == key and is_basic_pokemon(c) for c in st.hand):
            continue
        for c in st.deck:
            if norm(card_name(c)) == key and is_basic_pokemon(c):
                missing.append(c)
                seen.add(key)
                break
    return missing


def _ability_energy_cost_predicates(effect: Dict[str, Any]):
    preds = []
    for typ, _amount in ability_discard_energy_costs(effect):
        preds.append(lambda c, typ=typ: is_basic_typed_energy(c, typ))
    return preds


def has_search_discard_fodder_preserving_ability_cost(
    st: SimState,
    searcher: Dict[str, Any],
    effect: Dict[str, Any],
    target_norm: str,
    preserve_cards: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    cost = card_known_discard_cost(searcher)
    if cost <= 0:
        return True
    preserve_ids = {id(c) for c in (preserve_cards or [])}
    preserve_preds = _ability_energy_cost_predicates(effect)
    choices = []
    for c in st.hand:
        if c is searcher or id(c) in preserve_ids:
            continue
        if target_matches(c, target_norm):
            continue
        if any(pred(c) for pred in preserve_preds):
            continue
        choices.append(c)
    return len(choices) >= cost


def pay_search_discard_cost_preserving_ability_cost(
    st: SimState,
    searcher: Dict[str, Any],
    effect: Dict[str, Any],
    target_norm: str,
    stage: str,
    preserve_cards: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    cost = card_known_discard_cost(searcher)
    if cost <= 0:
        return True
    preserve_ids = {id(c) for c in (preserve_cards or [])}
    preserve_preds = _ability_energy_cost_predicates(effect)
    chosen = []
    for c in list(st.hand):
        if c is searcher or id(c) in preserve_ids:
            continue
        if target_matches(c, target_norm):
            continue
        if any(pred(c) for pred in preserve_preds):
            continue
        chosen.append(c)
        if len(chosen) >= cost:
            break
    if len(chosen) < cost:
        st.log.append({
            "event": "cannot_pay_search_cost_preserving_ability_cost",
            "card": card_name(searcher),
            "required_discards": cost,
            "stage": stage,
        })
        return False
    for c in chosen:
        st.hand.remove(c)
        st.discard.append(c)
    st.log.append({
        "event": "search_cost_discard",
        "card": card_name(searcher),
        "stage": stage,
        "discarded": [card_name(c) for c in chosen],
        "preserved_ability_energy_cost": bool(preserve_preds),
    })
    return True


def ability_requirement_chain_score(effect: Dict[str, Any], st: SimState, target_norm: str) -> float:
    """Score an ability after its missing requirement can be searched/benched."""
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0
    if ability_directly_searches_target(effect, target_norm, st.deck):
        return 8300.0
    draw_power = ability_draw_power(effect)
    if draw_power > 0:
        return 1500.0 * (1.0 - hypergeom_zero(max(1, len(st.deck)), target_remaining, min(draw_power, len(st.deck))))
    # Look/reorder abilities can matter when the target is already visible.
    for step in flatten_steps(effect):
        if step.get("op") in {"look_at_top_cards", "look_at_cards"}:
            n = amount_value(step.get("amount") or step.get("count") or step.get("number"), default=1)
            if any(target_matches(c, target_norm) for c in st.deck[:max(0, n)]):
                return 100.0
    return -1.0


def ability_requirement_search_candidates(st: SimState, target_norm: str, going: str, enable_chain_search: bool) -> List[Tuple[float, Dict[str, Any]]]:
    """Actions that use a search card to fetch a missing Pokémon requirement for an ability.

    This fixes the blindspot where a conditional ability is available in principle
    but needs a partner Pokémon first. Example:
      Lunatone in play + Ultra Ball + Basic Fighting Energy + 2 other cards
      -> Ultra Ball for Solrock -> bench Solrock -> Lunar Cycle -> draw 3.
    """
    out: List[Tuple[float, Dict[str, Any]]] = []
    potential_sources: List[Tuple[Dict[str, Any], bool]] = []  # (source, source_from_hand)
    for source in in_play_pokemon(st):
        potential_sources.append((source, False))
    for source in list(st.hand):
        if is_basic_pokemon(source) and not target_matches(source, target_norm) and len(st.bench) < bench_capacity(st):
            if card_has_usable_ability(source):
                potential_sources.append((source, True))

    for source, source_from_hand in potential_sources:
        for eff in ability_effects(source):
            key = ability_usage_key(source, eff)
            if key in st.abilities_used:
                continue
            name_n = norm(ability_name_from_effect(eff) + " " + ability_text_blob(eff))
            if "run errand" in name_n or "teal dance" in name_n or "last ditch" in name_n:
                continue
            if not ability_has_target_finding_ops(eff):
                continue
            missing = ability_missing_basic_requirement_cards_in_deck(eff, st, source)
            if not missing:
                continue
            # For now handle one searched requirement at a time. Multi-requirement abilities are rare;
            # the conservative choice is not to invent multiple searches in one virtual action.
            req = missing[0]
            for searcher in list(st.hand):
                if source_from_hand and searcher is source:
                    continue
                if not card_can_be_played_from_hand(searcher, going, st.supporter_used):
                    continue
                if not search_card_can_find_card(searcher, req):
                    continue
                preserve = [source] if source_from_hand else []
                if not has_search_discard_fodder_preserving_ability_cost(st, searcher, eff, target_norm, preserve):
                    continue
                score = ability_requirement_chain_score(eff, st, target_norm)
                if score <= 0:
                    continue
                # Slightly reward cards that search the requirement without using a Supporter.
                if not is_supporter(searcher):
                    score += 25.0
                out.append((score, {
                    "_virtual_action": "AbilityRequirementSearch",
                    "source": source,
                    "source_from_hand": source_from_hand,
                    "effect": eff,
                    "searcher": searcher,
                    "requirement": req,
                }))
    return out


def use_ability_requirement_search_chain(
    st: SimState,
    action: Dict[str, Any],
    rng: random.Random,
    target_norm: str,
    going: str,
    enable_chain_search: bool,
) -> None:
    source = action["source"]
    effect = action["effect"]
    searcher = action["searcher"]
    requirement = action["requirement"]
    ability_name = ability_name_from_effect(effect)
    stage = f"after_{card_name(searcher)}_for_{card_name(requirement)}_then_{ability_name}"

    # If the ability source was in hand, bench it first.
    if action.get("source_from_hand"):
        if source not in st.hand or len(st.bench) >= bench_capacity(st):
            return
        st.hand.remove(source)
        st.bench.append(source)
        st.actions_used += 1
        st.line.append(card_name(source))
        st.log.append({"event": "play_basic_to_bench", "card": card_name(source), "stage": stage, "purpose": "ability_source"})

    if searcher not in st.hand:
        return
    preserve = [source] if action.get("source_from_hand") else []
    if not pay_search_discard_cost_preserving_ability_cost(st, searcher, effect, target_norm, stage, preserve):
        return

    st.hand.remove(searcher)
    st.discard.append(searcher)
    if is_supporter(searcher):
        st.supporter_used = True
    st.actions_used += 1
    st.line.append(card_name(searcher))
    st.log.append({"event": "play_card", "card": card_name(searcher), "stage": stage, "purpose": "search_ability_requirement"})

    req = remove_first_matching(st.deck, lambda c: card_id(c) == card_id(requirement))
    if req is None:
        st.log.append({"event": "ability_requirement_search_failed", "card": card_name(requirement), "stage": stage})
        return
    st.hand.append(req)
    rng.shuffle(st.deck)
    st.log.append({"event": "search_deck_found_ability_requirement", "stage": stage, "searched_by": card_name(searcher), "selected": card_name(req)})

    # If the requirement itself is the target, the search already found it.
    if target_matches(req, target_norm):
        st.found = True
        st.found_stage = stage
        return

    if not bench_basic_by_name_from_hand(st, card_name(req), stage):
        return

    use_generic_ability(st, source, effect, rng, target_norm, going, enable_chain_search)

def target_on_top_via_ciphermaniac(st: SimState, target_norm: str, stage: str) -> bool:
    """Ciphermaniac puts target on top of deck, but does not itself find it."""
    target = remove_first_matching(st.deck, lambda c: target_matches(c, target_norm))
    if target is None:
        st.log.append({"event": "ciphermaniac_no_target_in_deck", "stage": stage})
        return False
    # The real card shuffles first, then puts chosen cards on top. For target
    # finding, all that matters is that the target becomes the top card.
    st.deck.insert(0, target)
    st.log.append({
        "event": "ciphermaniac_put_target_on_top",
        "stage": stage,
        "selected": card_name(target),
        "note": "Not a success until an immediate draw effect draws it.",
    })
    return True


def remove_basic_energies_for_crispin(st: SimState, stage: str) -> None:
    """Crispin deck-thinning approximation: remove up to 2 Basic Energy of different types.

    Crispin cannot find Ogerpon directly, but before Run Errand it can slightly
    improve the draw odds by thinning two Basic Energy cards. One would go to
    hand and one would attach, but neither is the target, so we model only the
    deck thinning for this target-finding scenario.
    """
    chosen = []
    used_types = set()
    for c in list(st.deck):
        if not is_energy(c):
            continue
        c_types = tuple(card_types(c)) or (card_name(c),)
        typ = c_types[0]
        if typ in used_types:
            continue
        st.deck.remove(c)
        chosen.append(card_name(c))
        used_types.add(typ)
        if len(chosen) >= 2:
            break
    if chosen:
        st.log.append({"event": "crispin_thin_basic_energy", "stage": stage, "removed_from_deck": chosen})


def draw_cards(st: SimState, n: int, stage: str) -> None:
    drawn = []
    for _ in range(max(0, n)):
        if not st.deck:
            break
        c = st.deck.pop(0)
        st.hand.append(c)
        drawn.append(card_name(c))
    if drawn:
        st.log.append({"event": "draw_cards", "stage": stage, "amount": len(drawn), "drawn": drawn})


# ---------------------------------------------------------------------
# TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION
# ---------------------------------------------------------------------
# Source-bound search legality guard.
#
# Why this exists:
# Earlier scoring correctly learned that Buddy-Buddy Poffin cannot directly
# search Basic Water Energy, but execution still had two permissive paths:
#   1. execute_steps() could select a target with filter_allows_card() alone.
#   2. choose_enabler_from_deck() treated "any card with a search effect" as a
#      valid chain enabler, even if the source could not search that enabler or
#      the enabler could not actually reach the target.
#
# This block makes all compiled search execution source-bound: a searched card
# must be reachable by the source card's own effect text/compiled search policy,
# and an enabler must itself have a real target path. This is deliberately broad
# and not Buddy-Buddy-specific.

def _turn1_v70_deck_with_candidate(deck, candidate):
    try:
        out = list(deck or [])
    except Exception:
        out = []
    if isinstance(candidate, dict) and not any(candidate is c for c in out):
        out.append(candidate)
    return out


def _turn1_v70_card_name_norm(card):
    try:
        return norm(card_name(card))
    except Exception:
        return ""


def _turn1_v70_find_card_by_name(st, name):
    name_n = norm(name or "")
    if not name_n:
        return None

    zones = []
    for attr in ("hand", "deck", "discard", "bench", "prizes"):
        try:
            value = getattr(st, attr, None)
            if isinstance(value, list):
                zones.append(value)
        except Exception:
            pass

    try:
        if getattr(st, "active", None) is not None:
            zones.append([st.active])
    except Exception:
        pass

    for zone in zones:
        for c in zone:
            if isinstance(c, dict) and _turn1_v70_card_name_norm(c) == name_n:
                return c
    return None


def _turn1_v70_current_source_names(st, stage=None):
    names = []

    try:
        log = list(getattr(st, "log", []) or [])
    except Exception:
        log = []

    # Prefer exact-stage log entries. Ability entries usually carry the actual
    # source Pokémon in the "source" field, while played cards carry "card".
    for ev in reversed(log):
        if not isinstance(ev, dict):
            continue
        if stage is not None and ev.get("stage") != stage:
            continue
        for key in ("source", "card", "searched_by"):
            val = ev.get(key)
            if val:
                names.append(str(val))

    # Fallback to recent log entries if recursion reused a nested stage.
    if not names:
        for ev in reversed(log[-8:]):
            if not isinstance(ev, dict):
                continue
            for key in ("source", "card", "searched_by"):
                val = ev.get(key)
                if val:
                    names.append(str(val))

    # Last fallback: line labels. This may be an ability name, not a card name,
    # so it is less reliable than the log source fields.
    try:
        if getattr(st, "line", None):
            names.append(str(st.line[-1]))
    except Exception:
        pass

    out = []
    seen = set()
    for name in names:
        n = norm(name)
        if n and n not in seen:
            out.append(name)
            seen.add(n)
    return out


def _turn1_v70_current_source_card(st, stage=None, source_card=None):
    if isinstance(source_card, dict):
        return source_card

    explicit = getattr(st, "_turn1_current_source_card", None)
    if isinstance(explicit, dict):
        return explicit

    for name in _turn1_v70_current_source_names(st, stage=stage):
        c = _turn1_v70_find_card_by_name(st, name)
        if isinstance(c, dict):
            return c
    return None


def _turn1_v70_source_can_fetch_candidate(source_card, candidate, deck):
    """True iff source_card can legally search/select candidate from deck."""
    if not isinstance(source_card, dict) or not isinstance(candidate, dict):
        return None

    # The card-level direct-search checker is the canonical broad policy. It
    # already handles source-text/HP/Bench restrictions from v67 and specific
    # fallbacks such as Ultra Ball, Fighting Gong, Energy Search, etc.
    candidate_deck = _turn1_v70_deck_with_candidate(deck, candidate)

    try:
        candidate_name = card_name(candidate)
        if candidate_name and card_directly_searches_target(source_card, norm(candidate_name), candidate_deck):
            return True
    except Exception:
        pass

    try:
        cid = card_id(candidate)
        if cid and card_directly_searches_target(source_card, norm(cid), candidate_deck):
            return True
    except Exception:
        pass

    return False


def _turn1_v70_source_can_select_target_card(st, filt, candidate, target_norm, step=None, stage=None, source_card=None):
    """Guard direct target selection during search execution."""
    if not isinstance(candidate, dict) or not target_matches(candidate, target_norm):
        return False

    src = _turn1_v70_current_source_card(st, stage=stage, source_card=source_card)
    if isinstance(src, dict):
        return bool(_turn1_v70_source_can_fetch_candidate(src, candidate, getattr(st, "deck", [])))

    # If source context is unavailable, preserve old behavior rather than
    # breaking unrelated legacy paths. Most real play/ability searches have log
    # source context, so this fallback should be rare.
    try:
        current_card_name = st.line[-1] if getattr(st, "line", None) else ""
        if norm(current_card_name) == "ultra ball":
            return card_supertype(candidate) == "Pokémon"
    except Exception:
        pass
    return filter_allows_card(filt, candidate)


def _turn1_v70_source_can_fetch_enabler(st, filt, candidate, target_norm, going, source_card=None, source_step=None, stage=None):
    """Guard chain-search enabler selection.

    The source must be able to search the enabler, and the enabler must have a
    real route toward the original target. A generic "has search" flag is not
    enough; that was the Buddy-Buddy -> Energy leak.
    """
    if not isinstance(candidate, dict):
        return False
    if not card_can_be_played_from_hand(candidate, going, st.supporter_used):
        return False

    src = _turn1_v70_current_source_card(st, stage=stage, source_card=source_card)
    if isinstance(src, dict):
        if not _turn1_v70_source_can_fetch_candidate(src, candidate, getattr(st, "deck", [])):
            return False
    else:
        if not filter_allows_card(filt, candidate):
            return False

    direct = bool(card_directly_searches_target(candidate, target_norm, _turn1_v70_deck_with_candidate(getattr(st, "deck", []), candidate)))
    draw_power = int(card_draw_power(candidate) or 0)

    # Evaluate special non-search play effects without allowing another generic
    # chain-search hop. This preserves valid enablers like Lillie's/Ciphermaniac
    # while excluding arbitrary search cards that do not reach the target.
    try:
        play_score_no_chain = float(score_playable_card(candidate, st, target_norm, going, False))
    except Exception:
        play_score_no_chain = 0.0

    return direct or draw_power > 0 or play_score_no_chain > 0

# END_TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION

def choose_enabler_from_deck(
    st: SimState,
    filt: Dict[str, Any],
    going: str,
    target_norm: str,
    source_card: Optional[Dict[str, Any]] = None,
    source_step: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION
    candidates = [
        c for c in st.deck
        if _turn1_v70_source_can_fetch_enabler(
            st,
            filt,
            c,
            target_norm,
            going,
            source_card=source_card,
            source_step=source_step,
            stage=stage,
        )
    ]
    if not candidates:
        return None

    def score(c: Dict[str, Any]) -> Tuple[int, int, float, str]:
        direct = 1 if card_directly_searches_target(c, target_norm, _turn1_v70_deck_with_candidate(st.deck, c)) else 0
        draw = int(card_draw_power(c) or 0)
        try:
            play_score = float(score_playable_card(c, st, target_norm, going, False))
        except Exception:
            play_score = 0.0
        return (direct, draw, play_score, card_name(c))

    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    if score(best)[:3] == (0, 0, 0.0):
        return None
    return best




# TURN1_EXACT_COMPILED_RUNTIME_V2
def _turn1_exact_amount_value(value, default=1):
    try:
        return int(amount_value(value, default=default))
    except Exception:
        try:
            return int(value)
        except Exception:
            return int(default)


def _turn1_exact_look_amount(step):
    look_at = step.get("look_at") if isinstance(step.get("look_at"), dict) else {}
    for value in (look_at.get("amount"), step.get("amount"), step.get("count"), step.get("number")):
        if value is None:
            continue
        n = _turn1_exact_amount_value(value, default=-1)
        if n >= 0:
            return n

    text = str(step.get("source_text") or step.get("text") or "")
    m = re.search(r"top\s+(\d+)\s+cards?", text, re.I)
    if m:
        return int(m.group(1))
    return 1


def _turn1_exact_choose_amount(step):
    selection = step.get("selection") if isinstance(step.get("selection"), dict) else {}
    for value in (
        step.get("amount"),
        step.get("count"),
        step.get("number"),
        selection.get("value"),
        selection.get("amount"),
        selection.get("count"),
    ):
        if value is None:
            continue
        n = _turn1_exact_amount_value(value, default=-1)
        if n >= 0:
            return n

    text = str(step.get("source_text") or step.get("text") or "")
    m = re.search(r"up to\s+(\d+)", text, re.I) or re.search(r"choose\s+(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    return 1


def _turn1_exact_step_filter(step):
    selection = step.get("selection") if isinstance(step.get("selection"), dict) else {}
    look_at = step.get("look_at") if isinstance(step.get("look_at"), dict) else {}

    for filt in (
        step.get("selection_filter"),
        selection.get("filter"),
        step.get("filter"),
        look_at.get("selection_filter"),
        look_at.get("filter"),
    ):
        if isinstance(filt, dict) and filt:
            return filt
    return {}


def _turn1_exact_norm(value):
    try:
        return norm(value)
    except Exception:
        return str(value or "").lower().strip()


def _turn1_exact_same_card(a, b):
    if a is b:
        return True
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    for key in ("_instance_id", "card_id", "representative_card_id"):
        av = a.get(key)
        bv = b.get(key)
        if av and bv and av == bv:
            return True

    try:
        return card_id(a) == card_id(b)
    except Exception:
        return False


def _turn1_exact_in_play_has_name(st, required_name):
    req = _turn1_exact_norm(required_name)
    if not req:
        return True

    try:
        pool = list(in_play_pokemon(st))
    except Exception:
        pool = []
        if getattr(st, "active", None) is not None:
            pool.append(st.active)
        pool.extend(getattr(st, "bench", []) or [])

    for c in pool:
        try:
            if _turn1_exact_norm(card_name(c)) == req:
                return True
        except Exception:
            pass
    return False


def _turn1_exact_required_discard_available(st, step):
    if step.get("op") not in {"discard_cards", "discard_card"}:
        return True

    required = bool(step.get("required_to_play")) or bool(step.get("cost"))
    if not required:
        return True

    selection = step.get("selection") if isinstance(step.get("selection"), dict) else {}
    filt = _turn1_exact_step_filter(step)
    n = _turn1_exact_amount_value(
        step.get("amount") or selection.get("value") or selection.get("amount") or step.get("count"),
        default=1,
    )

    if not filt:
        return len(getattr(st, "hand", []) or []) >= n

    matches = [c for c in getattr(st, "hand", []) or [] if filter_allows_card(filt, c)]
    return len(matches) >= n


def _turn1_exact_usage_key(step):
    group = step.get("group") or step.get("ability_name") or step.get("name") or step.get("source_text") or "compiled_effect"
    return "turn1_usage:" + _turn1_exact_norm(group)


def _turn1_exact_steps_preflight(st, steps, stage):
    for step in steps:
        if not isinstance(step, dict):
            continue

        op = step.get("op")

        if bool(step.get("requires_active_spot")):
            if getattr(st, "active", None) is None:
                st.log.append({"event": "blocked_requires_active_spot", "stage": stage})
                return False

        if op == "register_usage_limit":
            key = _turn1_exact_usage_key(step)
            limit = _turn1_exact_amount_value(step.get("limit") or step.get("max") or 1, default=1)
            if int(st.counts.get(key, 0) or 0) >= limit:
                st.log.append({"event": "blocked_usage_limit", "stage": stage, "usage_key": key, "limit": limit})
                return False

        if op == "play_condition":
            condition = step.get("condition") if isinstance(step.get("condition"), dict) else {}
            req = condition.get("requires_pokemon_in_play") if isinstance(condition.get("requires_pokemon_in_play"), dict) else None
            if req and not _turn1_exact_in_play_has_name(st, req.get("name")):
                st.log.append({"event": "blocked_play_condition", "stage": stage, "condition": condition})
                return False

        if op in {"discard_cards", "discard_card"} and not _turn1_exact_required_discard_available(st, step):
            st.log.append({"event": "blocked_missing_required_discard_cost", "stage": stage, "step": step})
            return False

    return True


def _turn1_exact_candidate_can_help(st, step, filt, candidate, target_norm, going, stage):
    if not isinstance(candidate, dict):
        return False

    try:
        if not filter_allows_card(filt, candidate):
            return False
    except Exception:
        return False

    try:
        if target_matches(candidate, target_norm):
            return True
    except Exception:
        pass

    # Legal intermediate enabler, e.g. Attract Customers can choose Lillie's,
    # then Lillie's may later draw into the target.
    try:
        if not card_can_be_played_from_hand(candidate, going, getattr(st, "supporter_used", False)):
            return False
    except Exception:
        return False

    try:
        deck_with_candidate = [candidate] + list(getattr(st, "deck", []) or [])
        if card_directly_searches_target(candidate, target_norm, deck_with_candidate):
            return True
    except Exception:
        pass

    try:
        if int(card_draw_power(candidate) or 0) > 0:
            return True
    except Exception:
        pass

    return False


def _turn1_exact_destination_zone(st, destination):
    dest = str(destination or "self.hand").lower()
    if "bench" in dest:
        return getattr(st, "bench", None)
    if "discard" in dest:
        return getattr(st, "discard", None)
    if "deck" in dest:
        return getattr(st, "deck", None)
    return getattr(st, "hand", None)


def _turn1_exact_remove_card_from_known_zones(st, card):
    for zone_name in ("deck", "hand", "discard", "bench", "prizes"):
        zone = getattr(st, zone_name, None)
        if not isinstance(zone, list):
            continue
        for i, c in enumerate(list(zone)):
            if _turn1_exact_same_card(c, card):
                zone.pop(i)
                return True
    return False


def _turn1_exact_move_cards(st, cards, destination, stage):
    dest_zone = _turn1_exact_destination_zone(st, destination)
    if dest_zone is None:
        return []

    moved = []
    for c in list(cards or []):
        _turn1_exact_remove_card_from_known_zones(st, c)
        dest_zone.append(c)
        moved.append(c)

    if moved:
        st.log.append({
            "event": "move_cards",
            "stage": stage,
            "destination": destination or "self.hand",
            "cards": [card_name(c) for c in moved],
            "exact_compiled_runtime": True,
        })

    return moved


def _turn1_exact_choose_from_pool(st, step, pool, target_norm, going, stage):
    filt = _turn1_exact_step_filter(step)
    amount = max(0, _turn1_exact_choose_amount(step))
    chosen = []

    for c in list(pool or []):
        if len(chosen) >= amount:
            break
        if _turn1_exact_candidate_can_help(st, step, filt, c, target_norm, going, stage):
            chosen.append(c)

    target_id = str(step.get("target_id") or step.get("cards_ref") or "chosen_cards")
    st.memory_cards[target_id] = list(chosen)

    if chosen:
        st.log.append({
            "event": "choose_cards",
            "stage": stage,
            "target_id": target_id,
            "source_ref": step.get("source_ref"),
            "selected": [card_name(c) for c in chosen],
            "filter": filt,
            "exact_compiled_runtime": True,
        })

    return chosen

def execute_steps(
    st: SimState,
    steps: Iterable[Dict[str, Any]],
    rng: random.Random,
    target_norm: str,
    going: str,
    stage: str,
    enable_chain_search: bool,
) -> None:
    steps = list(steps)
    if not _turn1_exact_steps_preflight(st, steps, stage):
        return
    # TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION
    for step in steps:
        if st.found:
            return
        op = step.get("op")

        if op in {"reference_global_rule", "register_usage_limit", "register_play_condition", "play_condition", "register_continuous_modifier", "register_trigger", "register_replacement_effect", "register_knockout_prize_rule"}:
            continue

        if op == "shuffle_deck":
            rng.shuffle(st.deck)
            st.log.append({"event": "shuffle_deck", "stage": stage})
            continue

        if op == "register_usage_limit":
            key = _turn1_exact_usage_key(step)
            st.counts[key] = int(st.counts.get(key, 0) or 0) + 1
            st.log.append({
                "event": "register_usage_limit",
                "stage": stage,
                "usage_key": key,
                "count": st.counts[key],
                "exact_compiled_runtime": True,
            })
            continue

        if op == "play_condition":
            continue

        if op in {"draw_cards", "draw_cards_per_coin_heads"}:
            n = draw_amount_from_step(step, counts=st.counts, coin_heads=st.coin_heads)
            if op == "draw_cards_per_coin_heads" and n == 0:
                n = st.coin_heads
            draw_cards(st, n, stage)
            if st.has_target_in_hand(target_norm):
                st.found = True
                st.found_stage = stage
            continue

        if op == "draw_until_hand_size":
            target_size = amount_value(step.get("target_hand_size"), default=len(st.hand), counts=st.counts)
            draw_cards(st, max(0, target_size - len(st.hand)), stage)
            if st.has_target_in_hand(target_norm):
                st.found = True
                st.found_stage = stage
            continue

        if op == "search_deck":
            filt = extract_filter(step)
            amt = search_amount(step)
            selected: List[Dict[str, Any]] = []
            source_card = _turn1_v70_current_source_card(st, stage=stage)

            def can_select_target(c: Dict[str, Any]) -> bool:
                return _turn1_v78_runtime_can_select_search_target(
                    st,
                    filt,
                    c,
                    target_norm,
                    step=step,
                    stage=stage,
                    source_card=source_card,
                )

            while len(selected) < amt:
                chosen = remove_first_matching(st.deck, can_select_target)
                if chosen is None:
                    break
                selected.append(chosen)

            if selected:
                st.hand.extend(selected)
                st.found = True
                st.found_stage = stage
                st.log.append({"event": "search_deck_found_target", "stage": stage, "selected": [card_name(c) for c in selected], "filter": filt, "source_bound_v70": True})
                continue

            # Optional chain search for a card that can draw/search into target.
            if enable_chain_search:
                enabler = choose_enabler_from_deck(
                    st,
                    filt,
                    going,
                    target_norm,
                    source_card=source_card,
                    source_step=step,
                    stage=stage,
                )
                if enabler is not None:
                    st.deck.remove(enabler)
                    st.hand.append(enabler)
                    st.log.append({"event": "search_deck_found_enabler", "stage": stage, "selected": card_name(enabler), "filter": filt, "source_bound_v70": True})
            continue

        if op in {"look_at_top_cards", "look_at_cards"}:
            n = _turn1_exact_look_amount(step)
            looked = st.deck[: max(0, n)]
            target_id = str(step.get("target_id") or "looked_cards")
            st.memory_cards[target_id] = list(looked)
            st.memory_cards["looked"] = list(looked)
            st.log.append({
                "event": "look_at_top_cards",
                "stage": stage,
                "target_id": target_id,
                "amount": len(looked),
                "cards": [card_name(c) for c in looked],
                "exact_compiled_runtime": True,
            })
            continue

        if op == "reorder_cards":
            looked = st.memory_cards.get("looked", [])
            if looked:
                target_seen = [c for c in looked if target_matches(c, target_norm)]
                others = [c for c in looked if not target_matches(c, target_norm)]
                new_top = target_seen + others
                st.deck[: len(looked)] = new_top
                st.log.append({"event": "reorder_cards", "stage": stage, "new_top": [card_name(c) for c in new_top]})
            continue

        if op in {"choose_cards", "put_card_into_hand", "move_card", "move_cards"}:
            if op == "choose_cards":
                source_ref = str(step.get("source_ref") or "")
                if source_ref and source_ref in st.memory_cards:
                    pool = list(st.memory_cards.get(source_ref) or [])
                elif "looked_cards" in st.memory_cards:
                    pool = list(st.memory_cards.get("looked_cards") or [])
                elif "looked" in st.memory_cards:
                    pool = list(st.memory_cards.get("looked") or [])
                else:
                    pool = list(st.deck)

                _turn1_exact_choose_from_pool(st, step, pool, target_norm, going, stage)
                continue

            cards_ref = str(step.get("cards_ref") or step.get("card_ref") or "")
            if cards_ref and cards_ref in st.memory_cards:
                selected = list(st.memory_cards.get(cards_ref) or [])
                moved = _turn1_exact_move_cards(st, selected, step.get("destination") or "self.hand", stage)
                if moved and st.has_target_in_hand(target_norm):
                    st.found = True
                    st.found_stage = stage
                continue

            filt = _turn1_exact_step_filter(step)
            destination = step.get("destination") or "self.hand"
            pool = list(st.deck)
            chosen = _turn1_exact_choose_from_pool(st, step, pool, target_norm, going, stage)
            moved = _turn1_exact_move_cards(st, chosen, destination, stage)
            if moved and st.has_target_in_hand(target_norm):
                st.found = True
                st.found_stage = stage
            continue

        if op in {"discard_cards", "discard_card"}:
            if int(st.counts.get("_skip_next_discard_cost", 0) or 0) > 0:
                st.counts["_skip_next_discard_cost"] = int(st.counts.get("_skip_next_discard_cost", 0) or 0) - 1
                st.log.append({"event": "skip_duplicate_discard_cost_step", "stage": stage})
                continue

            selection = step.get("selection") if isinstance(step.get("selection"), dict) else {}
            n = _turn1_exact_amount_value(
                step.get("amount") or selection.get("value") or selection.get("amount") or step.get("count"),
                default=1,
            )
            filt = _turn1_exact_step_filter(step)
            discarded = []

            for _ in range(max(0, n)):
                chosen = None

                for i, c in enumerate(list(st.hand)):
                    if filt and not filter_allows_card(filt, c):
                        continue
                    try:
                        if not filt and target_matches(c, target_norm):
                            continue
                    except Exception:
                        pass
                    chosen = st.hand.pop(i)
                    break

                if chosen is None:
                    for i, c in enumerate(list(st.hand)):
                        if filt and not filter_allows_card(filt, c):
                            continue
                        chosen = st.hand.pop(i)
                        break

                if chosen is None:
                    if step.get("required_to_play") or step.get("cost"):
                        st.log.append({
                            "event": "blocked_discard_cost_unpayable",
                            "stage": stage,
                            "filter": filt,
                            "required": n,
                        })
                        return
                    break

                st.discard.append(chosen)
                discarded.append(card_name(chosen))

            if discarded:
                st.log.append({
                    "event": "discard_cards",
                    "stage": stage,
                    "discarded": discarded,
                    "filter": filt,
                    "exact_compiled_runtime": True,
                })
            continue

        if op == "count_cards":
            count_id = str(step.get("count_id") or step.get("target_id") or "count")
            zone = str(step.get("from") or step.get("zone") or "self.hand")
            filt = extract_filter(step)
            pool = st.hand if "hand" in zone else st.deck if "deck" in zone else st.discard if "discard" in zone else []
            st.counts[count_id] = sum(1 for c in pool if filter_allows_card(filt, c))
            continue

        if op == "coin_flip":
            result = rng.choice(["heads", "tails"])
            st.coin_heads = 1 if result == "heads" else 0
            st.log.append({"event": "coin_flip", "stage": stage, "result": result})
            continue

        if op == "coin_flip_until":
            heads = 0
            for _ in range(20):
                if rng.choice([True, False]):
                    heads += 1
                else:
                    break
            st.coin_heads = heads
            st.log.append({"event": "coin_flip_until", "stage": stage, "heads": heads})
            continue

        if op == "branch_on_result":
            branch = None
            if st.coin_heads > 0:
                branch = step.get("heads") or step.get("on_heads") or step.get("if_heads")
            else:
                branch = step.get("tails") or step.get("on_tails") or step.get("if_tails")
            if branch is not None:
                execute_steps(st, list(flatten_steps(branch)), rng, target_norm, going, stage, enable_chain_search)
            continue

        if op in {"choose_yes_no", "branch_on_choice"}:
            branch = step.get("yes") or step.get("if_yes") or step.get("then") or step.get("steps")
            if branch is not None:
                execute_steps(st, list(flatten_steps(branch)), rng, target_norm, going, stage, enable_chain_search)
            continue

        if op in {"conditional", "conditional_effect"}:
            branch = step.get("then") or step.get("if_true") or step.get("steps")
            if branch is not None:
                execute_steps(st, list(flatten_steps(branch)), rng, target_norm, going, stage, enable_chain_search)
            continue

        if op in {
            "deal_attack_damage", "deal_damage", "modify_attack_damage", "set_attack_damage_from_coin_heads",
            "set_attack_damage_from_value", "set_attack_damage_per_damage_counter", "place_damage_counters",
            "heal_damage", "apply_special_condition", "remove_special_condition", "remove_special_conditions",
            "switch_active", "switch_active_with_bench", "choose_target", "choose_attack", "choose_player",
            "attach_card", "attach_cards", "attach_energy", "provide_energy", "move_energy", "discard_attached_energy",
            "evolve_pokemon", "evolve_pokemon_from_hand", "devolve_pokemon", "knock_out_pokemon",
            "attack_does_nothing", "attack_does_nothing_if", "conditional_attack_does_nothing",
            "ignore_resistance", "ignore_weakness_resistance", "ignore_effects_on_defending_pokemon",
            "ignore_defending_pokemon_damage_modifiers", "register_deck_construction_rule", "register_legality_note",
            "reveal_cards", "reveal_hand", "reveal_hand_to_player", "reveal_zone", "discard_stadium",
            "move_damage_counters", "distribute_damage_counters", "copy_and_use_attack",
            "move_pokemon_and_attached_cards", "play_trainer_as_pokemon", "grant_attack_from_attached_card",
        }:
            continue

        st.log.append({"event": "ignored_unknown_op", "op": op, "stage": stage, "step": step})


def score_playable_card(card: Dict[str, Any], st: SimState, target_norm: str, going: str, enable_chain_search: bool) -> float:
    # TURN1_V70_SOURCE_BOUND_SEARCH_EXECUTION
    if not card_can_be_played_from_hand(card, going, st.supporter_used):
        return -1.0
    if is_meowth_ex(card) and st.last_ditch_used:
        return -1.0
    if not has_enough_discard_fodder(st.hand, card, target_norm):
        return -1.0
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0

    # Direct target-to-hand lines. This is the authoritative fast path and
    # already carries v67 source-text restrictions.
    if card_directly_searches_target(card, target_norm, st.deck):
        if is_ultra_ball(card):
            return 10000.0
        if is_cyrano(card):
            return 9700.0
        return 9500.0

    if is_ciphermaniac(card):
        if going == "second" and not st.supporter_used and can_use_run_errand(st):
            return 9400.0
        return -1.0

    if is_meowth_ex(card):
        if st.last_ditch_used or going == "first" or st.supporter_used:
            return -1.0
        if has_card_in_zone(st.deck, is_cyrano) and target_is_pokemon_ex_in_pool(target_norm, st.deck):
            return 9300.0
        if can_use_run_errand(st) and has_card_in_zone(st.deck, is_ciphermaniac):
            return 9100.0
        if has_card_in_zone(st.deck, is_lillies_determination):
            return 4200.0
        return -1.0

    if is_lillies_determination(card):
        deck_size = max(1, len(st.deck) + len(st.hand) - 1)
        return 4500.0 * (1.0 - hypergeom_zero(deck_size, target_remaining, min(8, deck_size)))

    if is_crispin(card):
        if going == "second" and not st.supporter_used and can_use_run_errand(st):
            return 1200.0
        return -1.0

    deck_size = max(1, len(st.deck))
    draw_power = card_draw_power(card)
    draw_score = 1000.0 * (1.0 - hypergeom_zero(deck_size, target_remaining, min(draw_power, deck_size))) if draw_power else 0.0

    search_score = 0.0
    if enable_chain_search and card_has_search(card):
        for step in meaningful_steps(card):
            if step.get("op") == "search_deck":
                filt = extract_filter(step)
                if choose_enabler_from_deck(
                    st,
                    filt,
                    going,
                    target_norm,
                    source_card=card,
                    source_step=step,
                    stage=f"score_{card_name(card)}",
                ) is not None:
                    search_score = max(search_score, 250.0)

    look_score = 0.0
    for step in meaningful_steps(card):
        if step.get("op") in {"look_at_top_cards", "look_at_cards"}:
            n = amount_value(step.get("amount") or step.get("count") or step.get("number"), default=1)
            if any(target_matches(c, target_norm) for c in st.deck[:n]):
                look_score = max(look_score, 100.0)

    return max(draw_score, search_score, look_score)


def discard_fodder_cards(st: SimState, n: int, target_norm: str, stage: str) -> List[str]:
    """Discard low-value non-target cards for required costs.

    This does not try to solve the whole game; it only avoids discarding the
    target and prefers to throw away cards with no target-finding value.
    """
    discarded: List[str] = []
    for _ in range(max(0, n)):
        candidates = [c for c in st.hand if not target_matches(c, target_norm)]
        if not candidates:
            break

        def discard_priority(c: Dict[str, Any]) -> Tuple[int, int, int, str]:
            # Lower tuple is discarded first. Keep cards that can find/draw target.
            return (
                1 if card_has_specific_play_effect(c) or card_has_play_effect(c) else 0,
                1 if card_directly_searches_target(c, target_norm, st.deck) else 0,
                card_draw_power(c),
                card_name(c),
            )

        candidates.sort(key=discard_priority)
        chosen = candidates[0]
        st.hand.remove(chosen)
        st.discard.append(chosen)
        discarded.append(card_name(chosen))
    if discarded:
        st.log.append({"event": "discard_fodder", "stage": stage, "discarded": discarded})
    return discarded


def execute_specific_play_effect(
    st: SimState,
    card: Dict[str, Any],
    rng: random.Random,
    target_norm: str,
    going: str,
    stage: str,
    enable_chain_search: bool,
) -> bool:
    """Execute accurate narrow effects for relevant cards in the current deck.

    These are preferred over compiled generic steps when the generic compilation is
    too coarse for the target-finding question. In particular, Ciphermaniac puts
    cards on top of the deck, not into hand.
    """

    if is_ultra_ball(card):
        # Ultra Ball: discard 2 other cards, search your deck for a Pokémon,
        # reveal it, put it into your hand, then shuffle.
        discarded = discard_fodder_cards(st, 2, target_norm, stage)
        if len(discarded) < 2:
            st.log.append({"event": "ultra_ball_failed_missing_discard_fodder", "stage": stage})
            return True

        chosen = remove_first_matching(
            st.deck,
            lambda c: target_matches(c, target_norm) and card_supertype(c) == "Pokémon",
        )
        if chosen is not None:
            st.hand.append(chosen)
            st.found = True
            st.found_stage = stage
            st.log.append({
                "event": "search_deck_found_target",
                "stage": stage,
                "source": "Ultra Ball",
                "selected": [card_name(chosen)],
                "filter": {"supertype": "Pokémon"},
            })
        rng.shuffle(st.deck)
        st.log.append({"event": "shuffle_deck", "stage": stage, "source": "Ultra Ball"})
        return True

    if is_cyrano(card):
        # Cyrano: search for up to 3 Pokémon ex, reveal, put into hand.
        selected = []
        while len(selected) < 3:
            chosen = remove_first_matching(
                st.deck,
                lambda c: target_matches(c, target_norm) and is_pokemon_ex(c),
            )
            if chosen is None:
                break
            selected.append(chosen)
        if selected:
            st.hand.extend(selected)
            st.found = True
            st.found_stage = stage
            st.log.append({
                "event": "search_deck_found_target",
                "stage": stage,
                "source": "Cyrano",
                "selected": [card_name(c) for c in selected],
                "filter": {"supertype": "Pokémon", "subtype": "ex"},
            })
        rng.shuffle(st.deck)
        st.log.append({"event": "shuffle_deck", "stage": stage, "source": "Cyrano"})
        return True

    if is_ciphermaniac(card):
        # Ciphermaniac: search for 2 cards, shuffle, put them on top. This is
        # NOT success by itself. It becomes success if Run Errand immediately draws.
        rng.shuffle(st.deck)
        target_on_top_via_ciphermaniac(st, target_norm, stage)
        if can_use_run_errand(st):
            use_run_errand(st, target_norm, f"{stage}_then_Run_Errand")
        return True

    if is_lillies_determination(card):
        # Lillie's Determination: shuffle hand into deck, then draw 6; draw 8
        # instead if you have exactly 6 Prize cards remaining. On turn 1 we do.
        shuffled_back = [card_name(c) for c in st.hand]
        st.deck.extend(st.hand)
        st.hand.clear()
        rng.shuffle(st.deck)
        draw_n = 8 if len(st.prizes) == 6 else 6
        st.log.append({"event": "shuffle_hand_into_deck", "stage": stage, "cards": shuffled_back, "source": "Lillie's Determination"})
        draw_cards(st, draw_n, stage)
        if st.has_target_in_hand(target_norm):
            st.found = True
            st.found_stage = stage
            return True
        # If Mega Kangaskhan is active, optimally use Run Errand after the big draw.
        if can_use_run_errand(st):
            use_run_errand(st, target_norm, f"{stage}_then_Run_Errand")
        return True

    if is_crispin(card):
        # Crispin cannot find Ogerpon directly. It only matters here if it thins
        # Basic Energy before Run Errand.
        remove_basic_energies_for_crispin(st, stage)
        rng.shuffle(st.deck)
        if can_use_run_errand(st):
            use_run_errand(st, target_norm, f"{stage}_then_Run_Errand")
        return True

    return False


def play_card(st: SimState, card: Dict[str, Any], rng: random.Random, target_norm: str, going: str, enable_chain_search: bool) -> None:
    if card not in st.hand:
        return

    # Meowth ex — Last-Ditch Catch: when played from hand to Bench, search for
    # a Supporter. This is relevant going second because it can fetch Cyrano or
    # Ciphermaniac, then that Supporter can be played immediately.
    if is_meowth_ex(card):
        if st.last_ditch_used:
            return
        st.hand.remove(card)
        st.bench.append(card)
        st.last_ditch_used = True
        st.actions_used += 1
        st.line.append(card_name(card))
        stage = f"after_play_{card_name(card)}"
        st.log.append({"event": "play_basic_to_bench", "card": card_name(card), "stage": stage})

        if going == "second" and not st.supporter_used:
            # Priority: Cyrano direct search > Ciphermaniac + Run Errand > Lillie draw.
            supporter = None
            if target_is_pokemon_ex_in_pool(target_norm, st.deck):
                supporter = remove_first_matching(st.deck, is_cyrano)
            if supporter is None and can_use_run_errand(st):
                supporter = remove_first_matching(st.deck, is_ciphermaniac)
            if supporter is None:
                supporter = remove_first_matching(st.deck, is_lillies_determination)
            if supporter is not None:
                st.hand.append(supporter)
                rng.shuffle(st.deck)
                st.log.append({
                    "event": "last_ditch_catch_found_supporter",
                    "stage": stage,
                    "selected": card_name(supporter),
                })
                if card_can_be_played_from_hand(supporter, going, st.supporter_used):
                    play_card(st, supporter, rng, target_norm, going, enable_chain_search)
        return

    if not has_enough_discard_fodder(st.hand, card, target_norm):
        st.log.append({"event": "cannot_play_missing_discard_fodder", "card": card_name(card), "required_discards": card_known_discard_cost(card)})
        return
    st.hand.remove(card)
    st.discard.append(card)
    if is_supporter(card):
        st.supporter_used = True
    st.actions_used += 1
    action_name = card_name(card)
    st.line.append(action_name)
    stage = f"after_play_{action_name}"
    st.log.append({"event": "play_card", "card": action_name, "supporter_used": st.supporter_used})

    if execute_specific_play_effect(st, card, rng, target_norm, going, stage, enable_chain_search):
        return

    for eff in iter_effects(card):
        if effect_is_trivial_rule(eff):
            continue
        execute_steps(st, iter_steps(eff), rng, target_norm, going, stage=stage, enable_chain_search=enable_chain_search)
        if st.found:
            return


def simulate_one_trial(
    deck: List[Dict[str, Any]],
    rng: random.Random,
    target_norm: str,
    going: str,
    hand_size: int,
    prize_count: int,
    use_mulligans: bool,
    draw_for_turn: bool,
    max_actions: int,
    enable_chain_search: bool,
) -> Dict[str, Any]:
    mulligans = 0
    while True:
        shuffled = list(deck)
        rng.shuffle(shuffled)
        opening = shuffled[:hand_size]
        rest = shuffled[hand_size:]
        if not use_mulligans or any(is_basic_pokemon(c) for c in opening):
            break
        mulligans += 1
        if mulligans > 100:
            raise RuntimeError("Exceeded 100 mulligans in one trial; check Basic Pokémon count")

    prizes = rest[:prize_count]
    library = rest[prize_count:]

    target_copies_total = sum(1 for c in deck if target_matches(c, target_norm))
    target_copies_prized = sum(1 for c in prizes if target_matches(c, target_norm))
    target_copies_in_opening = sum(1 for c in opening if target_matches(c, target_norm))

    active = choose_optimal_active(opening, target_norm)
    hand_after_setup = list(opening)
    if active is not None and active in hand_after_setup:
        hand_after_setup.remove(active)

    st = SimState(deck=library, hand=hand_after_setup, prizes=list(prizes), active=active)
    if active is not None:
        st.log.append({"event": "choose_active", "active": card_name(active)})

    # Opening success means the target was in the opener / accessible at setup,
    # even if it was chosen as Active and is no longer literally in hand.
    if target_copies_in_opening > 0:
        st.found = True
        st.found_stage = "opening_hand"
    elif draw_for_turn:
        draw_cards(st, 1, "draw_for_turn")
        if st.has_target_in_hand(target_norm):
            st.found = True
            st.found_stage = "draw_for_turn"

    while not st.found and st.actions_used < max_actions:
        scored: List[Tuple[float, Any]] = []

        # Trainer / modeled-from-hand actions.
        playable = [c for c in list(st.hand) if card_can_be_played_from_hand(c, going, st.supporter_used)]
        for c in playable:
            score = score_playable_card(c, st, target_norm, going, enable_chain_search)
            if score > 0:
                scored.append((score, c))

        # Basic Pokémon from hand can be benched if their ability helps find the target.
        for c in list(st.hand):
            score = bench_basic_ability_score(st, c, target_norm, going)
            if score > 0:
                scored.append((score, {"_virtual_action": "BenchAbility", "card": c}))

        # Explicit active/bench abilities.
        ability_score = run_errand_score(st, target_norm)
        if ability_score > 0:
            scored.append((ability_score, {"_virtual_action": "Run Errand"}))

        td_score = teal_dance_score(st, target_norm)
        if td_score > 0:
            scored.append((td_score, {"_virtual_action": "Teal Dance"}))

        for score, source, eff in generic_ability_candidates(st, target_norm):
            scored.append((score, {"_virtual_action": "GenericAbility", "source": source, "effect": eff}))

        for score, action in ability_requirement_search_candidates(st, target_norm, going, enable_chain_search):
            scored.append((score, action))

        if not scored:
            break
        def _choice_label(x: Any) -> str:
            if isinstance(x, dict) and x.get("_virtual_action"):
                if x.get("card"):
                    return card_name(x["card"])
                if x.get("source"):
                    return card_name(x["source"])
                return str(x.get("_virtual_action"))
            return card_name(x) if isinstance(x, dict) else str(x)
        scored.sort(key=lambda x: (x[0], _choice_label(x[1])), reverse=True)
        _, chosen = scored[0]
        if isinstance(chosen, dict) and chosen.get("_virtual_action") == "Run Errand":
            use_run_errand(st, target_norm, "after_use_Run_Errand")
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "Teal Dance":
            use_teal_dance(st, target_norm, "after_use_Teal_Dance")
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "BenchAbility":
            bench_basic_for_ability(st, chosen["card"], rng, target_norm, going, enable_chain_search)
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "GenericAbility":
            use_generic_ability(st, chosen["source"], chosen["effect"], rng, target_norm, going, enable_chain_search)
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "AbilityRequirementSearch":
            use_ability_requirement_search_chain(st, chosen, rng, target_norm, going, enable_chain_search)
        else:
            play_card(st, chosen, rng, target_norm, going, enable_chain_search)

    return {
        "found": st.found,
        "found_stage": st.found_stage or "not_found",
        "line": " -> ".join(st.line) if st.line else "none",
        "mulligans": mulligans,
        "target_copies_total": target_copies_total,
        "target_copies_prized": target_copies_prized,
        "target_copies_in_opening": target_copies_in_opening,
        "all_target_copies_prized": target_copies_total > 0 and target_copies_prized == target_copies_total,
        "active": card_name(active) if active else None,
        "actions_used": st.actions_used,
        "final_hand_size": len(st.hand),
        "final_deck_size": len(st.deck),
        "log": st.log,
    }


def summarize_trials(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    found = sum(1 for r in results if r["found"])
    by_stage = Counter(r["found_stage"] for r in results)
    lines = Counter(r["line"] for r in results if r["found"] and r["line"] != "none")
    mulligans = Counter(r["mulligans"] for r in results)
    all_prized = sum(1 for r in results if r["all_target_copies_prized"])
    opening = sum(1 for r in results if r["found_stage"] == "opening_hand")
    draw_turn = sum(1 for r in results if r["found_stage"] == "draw_for_turn")
    actions = found - opening - draw_turn
    return {
        "trials": n,
        "successes": found,
        "probability": round(found / n, 6) if n else 0.0,
        "percent": pct(found / n) if n else 0.0,
        "ci95_percent": ci95(found, n),
        "found_in_opening_hand": {"successes": opening, "percent": pct(opening / n) if n else 0.0},
        "found_on_draw_for_turn": {"successes": draw_turn, "percent": pct(draw_turn / n) if n else 0.0},
        "found_after_actions": {"successes": actions, "percent": pct(actions / n) if n else 0.0},
        "all_target_copies_prized": {"trials": all_prized, "percent": pct(all_prized / n) if n else 0.0},
        "average_mulligans": round(sum(r["mulligans"] for r in results) / n, 6) if n else 0.0,
        "mulligan_distribution": [{"mulligans": k, "trials": v, "percent": pct(v / n)} for k, v in sorted(mulligans.items())],
        "found_stage_distribution": [{"stage": k, "trials": v, "percent": pct(v / n)} for k, v in by_stage.most_common()],
        "top_success_lines": [{"line": k, "count": v, "percent_of_trials": pct(v / n)} for k, v in lines.most_common(25)],
    }


# -----------------------------
# Line-audit helpers
# -----------------------------


def _log_events(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(result.get("log") or [])


def _event_matches(ev: Dict[str, Any], event: str, **conditions: Any) -> bool:
    if ev.get("event") != event:
        return False
    for k, v in conditions.items():
        if ev.get(k) != v:
            return False
    return True


def _has_event(log: Sequence[Dict[str, Any]], event: str, **conditions: Any) -> bool:
    return any(_event_matches(ev, event, **conditions) for ev in log)


def _discard_fodder_two_plus(log: Sequence[Dict[str, Any]]) -> bool:
    for ev in log:
        if ev.get("event") != "discard_fodder":
            continue
        discarded = ev.get("discarded") or []
        if isinstance(discarded, list) and len(discarded) >= 2:
            return True
    return False


def _line_contains(line: str, name: str) -> bool:
    return norm(name) in norm(line)


def audit_single_result(result: Dict[str, Any], going: str) -> Dict[str, Any]:
    """Audit whether one successful line is supported by concrete logged evidence.

    This is not a judge-level rules engine. It is a guardrail against over-counting
    lines like "Ciphermaniac" as success without the required immediate draw, or
    "Ultra Ball" without discard fodder. It checks the specific combos this script
    currently models.
    """
    line = str(result.get("line") or "none")
    log = _log_events(result)
    checks: Dict[str, bool] = {}
    notes: List[str] = []

    # Supporters are not legal on the first player's first turn in this model.
    supporter_names = ["Cyrano", "Ciphermaniac's Codebreaking", "Lillie's Determination", "Crispin"]
    uses_supporter = any(_line_contains(line, n) for n in supporter_names)
    if uses_supporter:
        checks["supporter_legal_for_turn"] = going == "second"
        if going != "second":
            notes.append("Supporter appeared in a going-first line.")

    if _line_contains(line, "Ultra Ball"):
        checks["ultra_ball_discarded_two"] = _discard_fodder_two_plus(log)
        checks["ultra_ball_searched_target"] = _has_event(log, "search_deck_found_target", source="Ultra Ball")
        if not checks["ultra_ball_discarded_two"]:
            notes.append("Ultra Ball line did not show a 2-card discard cost in the log.")
        if not checks["ultra_ball_searched_target"]:
            notes.append("Ultra Ball line did not show a target search in the log.")

    if _line_contains(line, "Run Errand"):
        checks["run_errand_active_kangaskhan"] = norm(result.get("active")) == norm("Mega Kangaskhan ex")
        checks["run_errand_used"] = _has_event(log, "use_ability", ability="Run Errand")
        if not checks["run_errand_active_kangaskhan"]:
            notes.append("Run Errand line did not start with Mega Kangaskhan ex Active.")
        if not checks["run_errand_used"]:
            notes.append("Run Errand line did not show the ability being used.")

    if _line_contains(line, "Cyrano"):
        checks["cyrano_searched_target"] = _has_event(log, "search_deck_found_target", source="Cyrano")
        if not checks["cyrano_searched_target"]:
            notes.append("Cyrano line did not show a target search.")

    if _line_contains(line, "Ciphermaniac"):
        checks["ciphermaniac_put_target_on_top"] = _has_event(log, "ciphermaniac_put_target_on_top")
        checks["ciphermaniac_followed_by_draw"] = _has_event(log, "use_ability", ability="Run Errand")
        if not checks["ciphermaniac_put_target_on_top"]:
            notes.append("Ciphermaniac line did not put target on top in the log.")
        if not checks["ciphermaniac_followed_by_draw"]:
            notes.append("Ciphermaniac line was not followed by an immediate draw effect.")

    if _line_contains(line, "Meowth ex"):
        checks["meowth_benched"] = _has_event(log, "play_basic_to_bench", card="Meowth ex")
        checks["meowth_found_supporter"] = _has_event(log, "last_ditch_catch_found_supporter")
        if not checks["meowth_benched"]:
            notes.append("Meowth ex line did not show Meowth being played to the Bench.")
        if not checks["meowth_found_supporter"]:
            notes.append("Meowth ex line did not show Last-Ditch Catch finding a Supporter.")

    if _line_contains(line, "Lillie's Determination"):
        checks["lillie_shuffled_hand_and_drew"] = _has_event(log, "shuffle_hand_into_deck", source="Lillie's Determination")
        if not checks["lillie_shuffled_hand_and_drew"]:
            notes.append("Lillie's Determination line did not show shuffle-hand-then-draw.")

    if _line_contains(line, "Teal Dance"):
        checks["teal_dance_used"] = _has_event(log, "use_ability", ability="Teal Dance")
        if not checks["teal_dance_used"]:
            notes.append("Teal Dance line did not show the ability being used.")

    if _line_contains(line, "Lunar Cycle"):
        checks["lunar_cycle_used"] = _has_event(log, "use_ability", ability="Lunar Cycle")
        checks["lunar_cycle_cost_paid"] = any(ev.get("event") == "ability_cost_discard" and ev.get("energy_type") == "Fighting" for ev in log)
        checks["lunar_cycle_solrock_requirement"] = any(
            ev.get("event") in {"ability_requirement_already_in_play", "ability_requirement_benched"} and norm(ev.get("card")) == norm("Solrock")
            for ev in log
        )
        if not checks["lunar_cycle_used"]:
            notes.append("Lunar Cycle line did not show the ability being used.")
        if not checks["lunar_cycle_cost_paid"]:
            notes.append("Lunar Cycle line did not show a Basic Fighting Energy discard cost.")
        if not checks["lunar_cycle_solrock_requirement"]:
            notes.append("Lunar Cycle line did not show Solrock in play or benched for the requirement.")

    if _line_contains(line, "Crispin"):
        # Crispin is only modeled as optional deck thinning before a draw. It can be legal
        # without being the final success event, so this check is informational.
        checks["crispin_line_has_draw_followup"] = _has_event(log, "use_ability", ability="Run Errand") or _line_contains(line, "Ultra Ball") or _line_contains(line, "Teal Dance")

    valid = all(checks.values()) if checks else True
    return {
        "line": line,
        "valid": valid,
        "checks": checks,
        "notes": notes,
    }


def build_line_audit(results: List[Dict[str, Any]], going: str, n_trials: int, examples_per_line: int = 2) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("found") and r.get("line") and r.get("line") != "none":
            grouped[str(r["line"])].append(r)

    rows: List[Dict[str, Any]] = []
    for line, rs in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        audits = [audit_single_result(r, going) for r in rs]
        valid_count = sum(1 for a in audits if a["valid"])
        check_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
        notes_counter: Counter[str] = Counter()
        for a in audits:
            for check_name, ok in a["checks"].items():
                check_totals[check_name]["pass" if ok else "fail"] += 1
            for note in a["notes"]:
                notes_counter[note] += 1

        examples = []
        for r, a in zip(rs, audits):
            examples.append({
                "found_stage": r.get("found_stage"),
                "active": r.get("active"),
                "actions_used": r.get("actions_used"),
                "final_hand_size": r.get("final_hand_size"),
                "final_deck_size": r.get("final_deck_size"),
                "audit_valid": a["valid"],
                "checks": a["checks"],
                "log": r.get("log", [])[:30],
            })
            if len(examples) >= examples_per_line:
                break

        rows.append({
            "line": line,
            "count": len(rs),
            "percent_of_trials": pct(len(rs) / n_trials) if n_trials else 0.0,
            "audit_valid_count": valid_count,
            "audit_invalid_count": len(rs) - valid_count,
            "verdict": "ok" if valid_count == len(rs) else "review",
            "checks": dict(check_totals),
            "notes": [{"note": k, "count": v} for k, v in notes_counter.most_common()],
            "examples": examples,
        })
    return rows


def write_line_audit_csv(path: str, scenarios: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "going", "line", "count", "percent_of_trials", "verdict", "audit_valid_count", "audit_invalid_count", "checks", "notes"],
        )
        writer.writeheader()
        for scenario in scenarios:
            for row in scenario.get("line_audit", []) or []:
                writer.writerow({
                    "scenario": scenario.get("scenario"),
                    "going": scenario.get("going"),
                    "line": row.get("line"),
                    "count": row.get("count"),
                    "percent_of_trials": row.get("percent_of_trials"),
                    "verdict": row.get("verdict"),
                    "audit_valid_count": row.get("audit_valid_count"),
                    "audit_invalid_count": row.get("audit_invalid_count"),
                    "checks": json.dumps(row.get("checks", {}), ensure_ascii=False),
                    "notes": json.dumps(row.get("notes", []), ensure_ascii=False),
                })


def target_finding_card_diagnostics(deck: List[Dict[str, Any]], target_norm: str) -> List[Dict[str, Any]]:
    """Summarize cards that the current policy thinks can help find the target."""
    rows = []
    unique_cards = []
    seen = set()
    for c in deck:
        cid = card_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        unique_cards.append(c)
    counts = Counter(card_id(c) for c in deck)
    for c in unique_cards:
        direct = card_directly_searches_target(c, target_norm, deck)
        draw = card_draw_power(c)
        has_search = card_has_search(c)
        cost = card_known_discard_cost(c)
        if not (direct or draw or has_search or cost):
            continue
        rows.append({
            "card_id": card_id(c),
            "name": card_name(c),
            "count": counts[card_id(c)],
            "supertype": card_supertype(c),
            "subtypes": card_subtypes(c),
            "is_supporter": is_supporter(c),
            "directly_searches_target": direct,
            "has_search": has_search,
            "draw_power_heuristic": draw,
            "known_discard_cost": cost,
            "playable_going_first_if_in_hand": card_can_be_played_from_hand(c, "first", supporter_used=False),
            "playable_going_second_if_in_hand": card_can_be_played_from_hand(c, "second", supporter_used=False),
        })
    rows.sort(key=lambda r: (r["directly_searches_target"], r["draw_power_heuristic"], r["count"], r["name"]), reverse=True)
    return rows


def write_summary_csv(path: str, scenario_summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "going", "trials", "successes", "percent", "ci95_low", "ci95_high", "opening_percent", "draw_for_turn_percent", "actions_percent", "all_prized_percent", "avg_mulligans"],
        )
        writer.writeheader()
        for s in scenario_summaries:
            sm = s["summary"]
            writer.writerow({
                "scenario": s["scenario"],
                "going": s["going"],
                "trials": sm["trials"],
                "successes": sm["successes"],
                "percent": sm["percent"],
                "ci95_low": sm["ci95_percent"]["low"],
                "ci95_high": sm["ci95_percent"]["high"],
                "opening_percent": sm["found_in_opening_hand"]["percent"],
                "draw_for_turn_percent": sm["found_on_draw_for_turn"]["percent"],
                "actions_percent": sm["found_after_actions"]["percent"],
                "all_prized_percent": sm["all_target_copies_prized"]["percent"],
                "avg_mulligans": sm["average_mulligans"],
            })


def run_scenario(args: argparse.Namespace, deck: List[Dict[str, Any]], going: str, target_norm: str) -> Dict[str, Any]:
    rng = random.Random(args.seed + (0 if going == "first" else 10_000_000))
    results = [
        simulate_one_trial(
            deck=deck,
            rng=rng,
            target_norm=target_norm,
            going=going,
            hand_size=args.hand_size,
            prize_count=args.prizes,
            use_mulligans=not args.no_mulligans,
            draw_for_turn=not args.no_draw_for_turn,
            max_actions=args.max_actions,
            enable_chain_search=args.chain_search,
        )
        for _ in range(args.trials)
    ]
    examples = []
    for r in results:
        if r["found"] and r["line"] != "none":
            examples.append({"line": r["line"], "found_stage": r["found_stage"], "log": r["log"][:20]})
        if len(examples) >= args.example_lines:
            break
    summary = summarize_trials(results)
    line_audit = build_line_audit(results, going, args.trials, examples_per_line=args.line_audit_examples)
    return {
        "scenario": f"turn1_find_{args.target_name}_{going}",
        "going": going,
        "summary": summary,
        "line_audit": line_audit,
        "examples": examples,
    }



def default_downloads_file(filename: str) -> str:
    """Return a sensible default output path in the user's Downloads folder.

    On Windows this becomes C:/Users/<user>/Downloads/<filename>.
    If Downloads cannot be found, fall back to the current working directory.
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.isdir(downloads):
        downloads = os.getcwd()
    return os.path.join(downloads, filename)


def write_result_json(path: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def print_compact_console_summary(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """Avoid dumping the giant JSON to PowerShell/terminal."""
    print("passed:", result.get("passed"))
    if result.get("error"):
        print("error:", result.get("error"))
    if result.get("warning"):
        print("warning:", result.get("warning"))

    deck_summary = result.get("deck_summary") or {}
    if deck_summary:
        print("deck_size:", deck_summary.get("deck_size"))
        print("target_copies:", deck_summary.get("target_copies"))
        print("basic_pokemon:", deck_summary.get("basic_pokemon"))

    unresolved = result.get("unresolved") or []
    print("unresolved:", unresolved)

    scenarios = result.get("scenarios") or []
    if scenarios:
        rows = []
        for scenario in scenarios:
            sm = scenario.get("summary", {})
            rows.append((
                scenario.get("going"),
                sm.get("percent"),
                (sm.get("ci95_percent") or {}).get("low"),
                (sm.get("ci95_percent") or {}).get("high"),
                (sm.get("found_in_opening_hand") or {}).get("percent"),
                (sm.get("found_on_draw_for_turn") or {}).get("percent"),
                (sm.get("found_after_actions") or {}).get("percent"),
            ))
        print("scenario_summary:", rows)

        print("top_success_lines:")
        for scenario in scenarios:
            sm = scenario.get("summary", {})
            print(" ", scenario.get("going"), (sm.get("top_success_lines") or [])[:10])

    print("full_json:", os.path.abspath(args.out))
    print("summary_csv:", os.path.abspath(args.csv_out))
    print("line_audit_csv:", os.path.abspath(args.line_audit_csv))

def main() -> None:
    ap = argparse.ArgumentParser(description="Turn 1 target-card finder with going-first/going-second restrictions and simple optimal play.")
    ap.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json", help="Compiled card JSON. Can be all compiled cards or complete-card seed.")
    ap.add_argument("--decklist", required=True, help="Decklist file: txt, csv, or json.")
    ap.add_argument("--target-name", required=True, help="Target card name/id substring to find by end of turn 1.")
    ap.add_argument("--going", choices=["first", "second", "both"], default="both")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hand-size", type=int, default=7)
    ap.add_argument("--prizes", type=int, default=6)
    ap.add_argument("--max-actions", type=int, default=20)
    ap.add_argument("--no-mulligans", action="store_true")
    ap.add_argument("--no-draw-for-turn", action="store_true")
    ap.add_argument("--complete-only", action="store_true", help="Use only parser-status complete cards for decklist resolution/effects.")
    ap.add_argument("--chain-search", action="store_true", help="Allow search cards to fetch another playable draw/search enabler if they cannot directly find the target.")
    ap.add_argument("--example-lines", type=int, default=5)
    ap.add_argument("--line-audit-examples", type=int, default=2, help="Number of detailed sample logs to keep per successful action line in the line audit.")
    ap.add_argument("--out", default=default_downloads_file("turn1_target_finder.json"), help="Full JSON report path. Defaults to Downloads.")
    ap.add_argument("--csv-out", default=default_downloads_file("turn1_target_finder_summary.csv"), help="Compact scenario summary CSV path. Defaults to Downloads.")
    ap.add_argument("--line-audit-csv", default=default_downloads_file("turn1_target_finder_line_audit.csv"), help="Line audit CSV path. Defaults to Downloads.")
    args = ap.parse_args()

    if load_compiled_cards is None:
        raise RuntimeError("Could not import tcgsim. Make sure this script is inside your project and src/tcgsim exists.")

    cards = load_compiled_cards(args.compiled)
    if args.complete_only:
        cards = filter_complete_cards(cards)

    raw_decklist = parse_decklist(args.decklist)
    deck, unresolved = resolve_decklist(raw_decklist, cards)
    target_norm = norm(args.target_name)

    result: Dict[str, Any] = {
        "passed": False,
        "compiled_source": args.compiled,
        "decklist_source": args.decklist,
        "target_name": args.target_name,
        "target_norm": target_norm,
        "trials": args.trials,
        "seed": args.seed,
        "assumptions": {
            "first_player_can_draw_for_turn": not args.no_draw_for_turn,
            "going_first_cannot_play_supporter_on_turn_1": True,
            "going_first_cannot_attack_on_turn_1": True,
            "going_second_can_play_supporter_on_turn_1": True,
            "focus": "Turn-1 target finding from hand/setup using relevant Trainer effects plus key Pokémon abilities for this deck.",
            "policy": "greedy best-first: direct target search > exact deck combos such as Ciphermaniac + Run Errand and Meowth -> Supporter > generic useful Pokémon Abilities > large draw > smaller draw/thinning.",
            "chain_search_enabled": args.chain_search,
            "basic_energy_proxies": "Enabled automatically when vanilla Basic Energy printings are absent from compiled JSON.",
            "line_audit": "Enabled. Combo lines such as Ultra Ball, Run Errand, Teal Dance, Meowth ex, and Ciphermaniac are audited against simulated logs for required evidence.",
            "exact_probability_integration": "Enabled. src/probability.py supplies exact legal-opening, natural draw, mulligan, and prize baselines; simulation estimates the conditional action layer.",
        },
        "decklist_entries": [{"count": c, "name": n} for c, n in raw_decklist],
        "unresolved": unresolved,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if unresolved:
        result["error"] = "Some decklist entries could not be resolved against the compiled card file."
        result["resolved_deck_size"] = len(deck)
        write_result_json(args.out, result)
        print_compact_console_summary(result, args)
        sys.exit(2)

    if len(deck) != 60:
        result["warning"] = f"Resolved deck has {len(deck)} cards, not 60. The script will still run, but probabilities may not represent a legal deck."

    target_copies = sum(1 for c in deck if target_matches(c, target_norm))
    basic_count = sum(1 for c in deck if is_basic_pokemon(c))
    result["deck_summary"] = {
        "deck_size": len(deck),
        "target_copies": target_copies,
        "basic_pokemon": basic_count,
        "energy": sum(1 for c in deck if is_energy(c)),
        "basic_energy_proxies": sum(1 for c in deck if c.get("parser_status") == "proxy_basic_energy"),
        "trainer": sum(1 for c in deck if is_trainer(c)),
        "top_cards": [{"name": k, "count": v} for k, v in Counter(card_name(c) for c in deck).most_common(25)],
    }
    result["target_finding_card_diagnostics"] = target_finding_card_diagnostics(deck, target_norm)
    result["raw_hypergeometric_reference"] = {
        "note": "Reference only. These are unconditioned raw 60-card calculations; prefer exact_legal_hand_baselines below for game-start probabilities.",
        "opening_hand_has_target_percent": pct(hypergeom_at_least_one(len(deck), target_copies, args.hand_size)),
        "opening_hand_has_no_basic_percent": pct(hypergeom_zero(len(deck), basic_count, args.hand_size)),
        "at_least_one_target_prized_unconditional_percent": pct(hypergeom_at_least_one(len(deck), target_copies, args.prizes)),
        "all_target_copies_prized_unconditional_percent": pct((ncr(target_copies, target_copies) * ncr(len(deck) - target_copies, args.prizes - target_copies) / ncr(len(deck), args.prizes)) if 0 < target_copies <= args.prizes <= len(deck) else 0.0),
    }
    result["exact_legal_hand_baselines"] = build_exact_probability_baselines(
        deck=deck,
        target_norm=target_norm,
        hand_size=args.hand_size,
        prize_count=args.prizes,
        draw_for_turn=not args.no_draw_for_turn,
        max_mulligans=6,
    )

    if target_copies <= 0:
        result["error"] = "Target card was not found in the resolved decklist."
        write_result_json(args.out, result)
        print_compact_console_summary(result, args)
        sys.exit(3)

    goings = ["first", "second"] if args.going == "both" else [args.going]
    scenarios = [run_scenario(args, deck, going, target_norm) for going in goings]
    add_exact_plus_simulation_fields(scenarios, result.get("exact_legal_hand_baselines", {}))
    result["scenarios"] = scenarios
    result["passed"] = True

    write_result_json(args.out, result)
    write_summary_csv(args.csv_out, scenarios)
    write_line_audit_csv(args.line_audit_csv, scenarios)
    print_compact_console_summary(result, args)




# -----------------------------------------------------------------------------
# TURN1_COMPILED_EFFECT_RUNTIME_V39
# -----------------------------------------------------------------------------
# Root fix:
# Use the compiled effect/search filters and printed/source text to decide:
# - what a search effect can actually find
# - whether an Ability is available from Active / Bench / when-played-from-hand
# - whether a Basic Pokemon should be chosen Active at setup for an active-only
#   ability that can satisfy the current target
#
# This is intentionally generic. It is not a Chien-Pao/Lumineon special case.
#
# Conservative retreat policy:
# This patch does NOT assume a free retreat. Active-only abilities are usable only
# when the source Pokemon is actually Active. A later board-action layer can model
# attach-energy + retreat-cost + once-per-turn retreat, but until then this patch
# prevents illegal active-only Ability lines instead of inventing them.


def _turn1_v39_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_v39_card_name(card):
    try:
        return card_name(card)
    except Exception:
        if isinstance(card, dict):
            ident = card.get("identity") or {}
            return (
                card.get("name")
                or card.get("card_name")
                or ident.get("name")
                or ident.get("canonical_name")
                or ""
            )
        return ""


def _turn1_v39_card_supertype(card):
    try:
        return card_supertype(card)
    except Exception:
        if isinstance(card, dict):
            ident = card.get("identity") or {}
            return str(card.get("supertype") or ident.get("supertype") or "")
        return ""


def _turn1_v39_card_subtypes(card):
    try:
        return list(card_subtypes(card))
    except Exception:
        vals = []
        if isinstance(card, dict):
            ident = card.get("identity") or {}
            for obj in (card, ident):
                for key in ("subtype", "subtypes", "trainerType"):
                    v = obj.get(key)
                    if isinstance(v, list):
                        vals.extend(str(x) for x in v)
                    elif v:
                        vals.append(str(v))
        return vals


def _turn1_v39_card_types(card):
    try:
        return list(card_types(card))
    except Exception:
        vals = []
        if isinstance(card, dict):
            ident = card.get("identity") or {}
            for obj in (card, ident):
                v = obj.get("types") or obj.get("type")
                if isinstance(v, list):
                    vals.extend(str(x) for x in v)
                elif v:
                    vals.append(str(v))
        return vals


def _turn1_v39_card_hp(card):
    import re as _re

    candidates = []
    if isinstance(card, dict):
        candidates.extend([card.get("hp"), card.get("raw_hp")])
        for key in ("identity", "gameplay", "raw_card", "source"):
            obj = card.get(key)
            if isinstance(obj, dict):
                candidates.extend([obj.get("hp"), obj.get("raw_hp")])

    for value in candidates:
        if value is None:
            continue
        m = _re.search(r"\d+", str(value))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    return None


_TURN1_V39_TYPES = {
    "grass": ["grass", "g"],
    "fire": ["fire", "r"],
    "water": ["water", "w"],
    "lightning": ["lightning", "electric", "l"],
    "psychic": ["psychic", "p"],
    "fighting": ["fighting", "f"],
    "darkness": ["darkness", "dark", "d"],
    "metal": ["metal", "steel", "m"],
    "colorless": ["colorless", "c"],
}


def _turn1_v39_is_pokemon(card):
    return _turn1_v39_norm(_turn1_v39_card_supertype(card)) in {"pokemon", "pokémon"}


def _turn1_v39_is_trainer(card):
    return _turn1_v39_norm(_turn1_v39_card_supertype(card)) == "trainer"


def _turn1_v39_is_energy(card):
    try:
        return is_energy(card)
    except Exception:
        return _turn1_v39_norm(_turn1_v39_card_supertype(card)) == "energy"


def _turn1_v39_is_basic_pokemon(card):
    try:
        return is_basic_pokemon(card)
    except Exception:
        return _turn1_v39_is_pokemon(card) and "basic" in {_turn1_v39_norm(x) for x in _turn1_v39_card_subtypes(card)}


def _turn1_v39_is_basic_energy(card):
    if not _turn1_v39_is_energy(card):
        return False
    name_n = _turn1_v39_norm(_turn1_v39_card_name(card))
    subs = {_turn1_v39_norm(x) for x in _turn1_v39_card_subtypes(card)}
    return "basic" in subs or name_n.startswith("basic ")


def _turn1_v39_has_type(card, typ):
    typ_n = _turn1_v39_norm(typ)
    name_n = _turn1_v39_norm(_turn1_v39_card_name(card))
    types = {_turn1_v39_norm(x) for x in _turn1_v39_card_types(card)}
    aliases = _TURN1_V39_TYPES.get(typ_n, [typ_n])
    return any(_turn1_v39_norm(a) in types or _turn1_v39_norm(a) in name_n for a in aliases)


def _turn1_v39_trainer_kind(card, kind):
    if not _turn1_v39_is_trainer(card):
        return False
    kind_n = _turn1_v39_norm(kind)
    name_n = _turn1_v39_norm(_turn1_v39_card_name(card))
    subs = {_turn1_v39_norm(x) for x in _turn1_v39_card_subtypes(card)}
    return kind_n in subs or kind_n in name_n


def _turn1_v39_is_pokemon_ex(card):
    if not _turn1_v39_is_pokemon(card):
        return False
    name_n = _turn1_v39_norm(_turn1_v39_card_name(card))
    subs = {_turn1_v39_norm(x) for x in _turn1_v39_card_subtypes(card)}
    return name_n.endswith(" ex") or " ex " in f" {name_n} " or "ex" in subs


def _turn1_v39_hp_ok(card, target_text):
    import re as _re

    t = _turn1_v39_norm(target_text)
    m = _re.search(r"(\d+)\s*hp\s*or\s*less", t)
    if not m:
        return True
    hp = _turn1_v39_card_hp(card)
    if hp is None:
        return False
    try:
        return hp <= int(m.group(1))
    except Exception:
        return False


def _turn1_v39_phrase_has_typed_target(phrase, typ, noun):
    p = _turn1_v39_norm(phrase)
    for alias in _TURN1_V39_TYPES.get(typ, [typ]):
        a = _turn1_v39_norm(alias)
        if f"{a} {noun}" in p:
            return True
    return False


def _turn1_v39_search_target_phrase(blob):
    """Extract what the effect searches/chooses/reveals, not ability conditions."""
    import re as _re

    b = _turn1_v39_norm(blob)
    patterns = [
        r"search (?:your|the) deck(?: and (?:your )?discard pile)? for (.*?)(?:, reveal| reveal| and reveal|,? and put|,? put| then shuffle| shuffle|\.|$)",
        r"look at the top \d+ cards? of your deck.*?reveal (.*?)(?: card| cards|,| and put| put|$)",
        r"choose (.*?)(?: from (?:your )?deck| from among them|,| and put| put|$)",
    ]
    for pat in patterns:
        m = _re.search(pat, b)
        if not m:
            continue
        phrase = m.group(1)
        phrase = _re.sub(r"^(up to|exactly)?\s*\d+\s+", "", phrase).strip()
        phrase = _re.sub(r"^(a|an|any|one|two)\s+", "", phrase).strip()
        if phrase:
            return phrase
    return b


def _turn1_v39_filter_raw_decision(filt, card):
    if not isinstance(filt, dict):
        return None
    blob = filter_text_blob(filt)
    if not blob:
        return None

    blob_n = _turn1_v39_norm(blob)
    has_raw = any(k in filt for k in ("raw_text", "source_text", "text"))
    has_search_language = any(
        p in blob_n
        for p in (
            "search your deck",
            "look at the top",
            "reveal",
            "choose",
            "put them into your hand",
            "put it into your hand",
            "put them onto your bench",
            "put it onto your bench",
        )
    )
    if not has_raw and not has_search_language:
        return None

    target = _turn1_v39_search_target_phrase(blob)
    t = _turn1_v39_norm(target)
    tests = []

    # Energy target filters.
    for typ in _TURN1_V39_TYPES:
        if _turn1_v39_phrase_has_typed_target(t, typ, "energy"):
            if "basic" in t:
                tests.append(lambda c, typ=typ: _turn1_v39_is_basic_energy(c) and _turn1_v39_has_type(c, typ))
            else:
                tests.append(lambda c, typ=typ: _turn1_v39_is_energy(c) and _turn1_v39_has_type(c, typ))
    if "basic energy" in t:
        tests.append(lambda c: _turn1_v39_is_basic_energy(c))
    elif "energy" in t and not tests:
        tests.append(lambda c: _turn1_v39_is_energy(c))

    # Pokemon target filters.
    for typ in _TURN1_V39_TYPES:
        if _turn1_v39_phrase_has_typed_target(t, typ, "pokemon") or _turn1_v39_phrase_has_typed_target(t, typ, "pokémon"):
            if "basic" in t:
                tests.append(lambda c, typ=typ, target=t: _turn1_v39_is_basic_pokemon(c) and _turn1_v39_has_type(c, typ) and _turn1_v39_hp_ok(c, target))
            else:
                tests.append(lambda c, typ=typ, target=t: _turn1_v39_is_pokemon(c) and _turn1_v39_has_type(c, typ) and _turn1_v39_hp_ok(c, target))
    if "pokemon ex" in t or "pokémon ex" in t:
        tests.append(lambda c: _turn1_v39_is_pokemon_ex(c))
    if "basic pokemon" in t or "basic pokémon" in t:
        tests.append(lambda c, target=t: _turn1_v39_is_basic_pokemon(c) and _turn1_v39_hp_ok(c, target))
    elif ("pokemon" in t or "pokémon" in t) and not any(x in t for x in ("energy", "supporter", "item", "stadium", "tool")):
        tests.append(lambda c, target=t: _turn1_v39_is_pokemon(c) and _turn1_v39_hp_ok(c, target))

    # Trainer target filters.
    if "supporter" in t:
        tests.append(lambda c: _turn1_v39_trainer_kind(c, "supporter"))
    if "item" in t:
        tests.append(lambda c: _turn1_v39_trainer_kind(c, "item"))
    if "stadium" in t:
        tests.append(lambda c: _turn1_v39_trainer_kind(c, "stadium"))
    if "tool" in t:
        tests.append(lambda c: _turn1_v39_trainer_kind(c, "tool") or _turn1_v39_trainer_kind(c, "pokemon tool"))
    if "trainer" in t and not any(x in t for x in ("supporter", "item", "stadium", "tool")):
        tests.append(lambda c: _turn1_v39_is_trainer(c))

    if not tests and ("any card" in t or t in {"card", "a card"}):
        return True
    if tests:
        return any(test(card) for test in tests)
    if has_search_language:
        # Conservative: a compiled/source search filter with unclassified target
        # cannot search arbitrary cards.
        return False
    return None


_ORIG_FILTER_ALLOWS_CARD_V39 = filter_allows_card


def filter_allows_card(filt, card):
    decision = _turn1_v39_filter_raw_decision(filt, card)
    if decision is not None:
        return bool(decision)
    return _ORIG_FILTER_ALLOWS_CARD_V39(filt, card)


# --------------------------
# Ability lifecycle handling
# --------------------------

def _turn1_v39_effect_text(effect):
    try:
        return ability_text_blob(effect)
    except Exception:
        return str(effect or "")


def _turn1_v39_effect_blob(effect):
    return _turn1_v39_norm(_turn1_v39_effect_text(effect))


def _turn1_v39_effect_requires_active(effect):
    b = _turn1_v39_effect_blob(effect)
    active_phrases = [
        "if this pokemon is in the active spot",
        "if this pokémon is in the active spot",
        "while this pokemon is in the active spot",
        "while this pokémon is in the active spot",
        "as long as this pokemon is in the active spot",
        "as long as this pokémon is in the active spot",
        "this pokemon is in the active spot",
        "this pokémon is in the active spot",
        "your active pokemon",
        "your active pokémon",
    ]
    return any(p in b for p in active_phrases)


def _turn1_v39_effect_is_when_played_to_bench(effect):
    b = _turn1_v39_effect_blob(effect)
    return (
        "when you play this pokemon from your hand onto your bench" in b
        or "when you play this pokémon from your hand onto your bench" in b
        or "when you play this card from your hand onto your bench" in b
    )


def _turn1_v39_effect_has_usable_search_or_draw(effect, target_norm, deck):
    # Direct search must be allowed by compiled/source filters.
    try:
        if ability_directly_searches_target(effect, target_norm, deck):
            return True
    except Exception:
        pass
    # Generic draw/look remains a valid dig action if it can see/draw the target.
    try:
        return ability_draw_power(effect) > 0 or any(
            s.get("op") in {"look_at_top_cards", "look_at_cards", "reorder_cards"}
            for s in flatten_steps(effect)
        )
    except Exception:
        return False


def _turn1_v39_ability_context_ok(effect, st, source, target_norm, context):
    # context: "in_play" for normal reusable/checkable abilities, "just_benched" after playing source from hand.
    if _turn1_v39_effect_requires_active(effect) and source is not getattr(st, "active", None):
        return False
    if context == "in_play" and _turn1_v39_effect_is_when_played_to_bench(effect):
        return False
    if context == "just_benched" and _turn1_v39_effect_requires_active(effect):
        return False
    try:
        if not ability_requirements_can_be_met(effect, st, source):
            return False
        if not ability_costs_can_be_paid(effect, st, target_norm):
            return False
    except Exception:
        return False
    return _turn1_v39_effect_has_usable_search_or_draw(effect, target_norm, st.deck)


_ORIG_ABILITY_READY_FOR_TARGET_FINDING_V39 = ability_ready_for_target_finding


def ability_ready_for_target_finding(effect, st, source, target_norm):
    # Normal in-play ability use: no bench-trigger abilities, active-only only from Active.
    if not _turn1_v39_ability_context_ok(effect, st, source, target_norm, "in_play"):
        return False
    return True


_ORIG_SCORE_GENERIC_ABILITY_V39 = score_generic_ability


def score_generic_ability(st, source, effect, target_norm):
    if not ability_ready_for_target_finding(effect, st, source, target_norm):
        return -1.0
    return _ORIG_SCORE_GENERIC_ABILITY_V39(st, source, effect, target_norm)


def _turn1_v39_score_ability_context(st, source, effect, target_norm, context):
    if not _turn1_v39_ability_context_ok(effect, st, source, target_norm, context):
        return -1.0
    if ability_directly_searches_target(effect, target_norm, st.deck):
        return 8800.0
    target_remaining = sum(1 for c in st.deck if target_matches(c, target_norm))
    if target_remaining <= 0 or not st.deck:
        return -1.0
    draw_power = ability_draw_power(effect)
    if draw_power:
        return 1000.0 * (1.0 - hypergeom_zero(len(st.deck), target_remaining, min(draw_power, len(st.deck))))
    return -1.0


_ORIG_CAN_BENCH_BASIC_FOR_ABILITY_V39 = can_bench_basic_for_ability


def can_bench_basic_for_ability(st, card, target_norm):
    if not is_basic_pokemon(card):
        return False
    if len(st.bench) >= bench_capacity(st):
        return False
    if card not in st.hand:
        return False
    if target_matches(card, target_norm):
        return False
    for eff in ability_effects(card):
        if _turn1_v39_score_ability_context(st, card, eff, target_norm, "just_benched") > 0:
            return True
    return False


_ORIG_BENCH_BASIC_ABILITY_SCORE_V39 = bench_basic_ability_score


def bench_basic_ability_score(st, card, target_norm, going):
    if not can_bench_basic_for_ability(st, card, target_norm):
        return -1.0
    if is_meowth_ex(card):
        return score_playable_card(card, st, target_norm, going, True)
    best = -1.0
    for eff in ability_effects(card):
        best = max(best, _turn1_v39_score_ability_context(st, card, eff, target_norm, "just_benched"))
    return best


_ORIG_BENCH_BASIC_FOR_ABILITY_V39 = bench_basic_for_ability


def bench_basic_for_ability(st, card, rng, target_norm, going, enable_chain_search):
    if not can_bench_basic_for_ability(st, card, target_norm):
        return
    if is_meowth_ex(card):
        play_card(st, card, rng, target_norm, going, enable_chain_search)
        return
    st.hand.remove(card)
    st.bench.append(card)
    st.actions_used += 1
    st.line.append(card_name(card))
    stage = f"after_bench_{card_name(card)}"
    st.log.append({"event": "play_basic_to_bench", "card": card_name(card), "stage": stage})
    candidates = []
    for eff in ability_effects(card):
        score = _turn1_v39_score_ability_context(st, card, eff, target_norm, "just_benched")
        if score > 0:
            candidates.append((score, eff))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, eff = candidates[0]
        use_generic_ability(st, card, eff, rng, target_norm, going, enable_chain_search)


_ORIG_CHOOSE_OPTIMAL_ACTIVE_V39 = choose_optimal_active


def choose_optimal_active(opening, target_norm):
    basics = [c for c in opening if is_basic_pokemon(c)]
    if not basics:
        return None

    # Prefer a Basic whose active-only compiled/source Ability can directly help this target.
    for c in basics:
        fake_rest = [x for x in opening if x is not c]
        st = SimState(deck=list(fake_rest), hand=list(fake_rest), prizes=[], active=c)
        # Use the opening as a small proxy for known cards; this only decides setup Active.
        for eff in ability_effects(c):
            if not _turn1_v39_effect_requires_active(eff):
                continue
            # For active choice, require the effect target class to match; exact deck copies are checked later.
            if _turn1_v39_effect_has_usable_search_or_draw(eff, target_norm, list(opening)):
                return c

    return _ORIG_CHOOSE_OPTIMAL_ACTIVE_V39(opening, target_norm)


if __name__ == "__main__":
    main()



# ---------------------------------------------------------------------
# TURN1_BLOCK_OPPONENT_DEPENDENT_ACCESS_V40
# ---------------------------------------------------------------------
# Broad rule:
# Cards whose access value depends on the opponent's board/hand/deck are not
# valid deterministic Turn-1 consistency tools unless the simulator explicitly
# models that opponent state.
#
# Example:
#   Morty's Conviction:
#     "Draw a card for each of your opponent's Benched Pokemon."
#
# Since this simulator does not currently model opponent bench size, that effect
# must not be credited as generic draw/access.


# ---------------------------------------------------------------------
# TURN1_DIRECT_TARGETFINDER_V40_CACHE_V54
# ---------------------------------------------------------------------
# Direct hot-path fix, not a same-name wrapper.
#
# The profiler showed play_card spending most of its time in the v40
# opponent-dependent-access guard. That guard repeatedly normalized and
# flattened the same card dictionaries 100k+ times during chain-search.
#
# This block gives the original v40 helper functions their own caches and
# replaces their bodies directly below.

_TURN1_V54_NORM_CACHE = {}
_TURN1_V54_OPP_DEP_ACCESS_CACHE = {}


def _turn1_v54_card_cache_key(card):
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        key = (
            card.get("card_id")
            or card.get("representative_card_id")
            or card.get("id")
            or ident.get("card_id")
            or ident.get("id")
            or ident.get("canonical_id")
        )
        if key:
            return ("id", str(key))
        name = card.get("name") or card.get("card_name") or ident.get("name") or ident.get("canonical_name")
        set_code = card.get("set_code") or ident.get("set_code") or card.get("set") or ident.get("set")
        number = card.get("number") or card.get("collector_number") or ident.get("number") or ident.get("collector_number")
        if name:
            return ("name", str(name), str(set_code or ""), str(number or ""))
    if isinstance(card, str):
        return ("str", card)
    return ("obj", id(card))

def _turn1_v40_norm(value):
    """Cached normalization for the v40 opponent-dependent-access guard."""
    import re as _re
    import unicodedata as _unicodedata

    try:
        key = _turn1_v54_card_cache_key(value)
        cached = _TURN1_V54_NORM_CACHE.get(key)
        if cached is not None:
            return cached

        if isinstance(value, str):
            blob = value
        else:
            try:
                blob = _turn1_v40_flatten_strings(value)
            except Exception:
                blob = str(value or "")

        s = _unicodedata.normalize("NFKD", str(blob or ""))
        s = "".join(ch for ch in s if not _unicodedata.combining(ch))
        s = s.lower().replace("’", "'").replace("`", "'")
        s = _re.sub(r"\s+", " ", s).strip()

        if len(_TURN1_V54_NORM_CACHE) < 250000:
            _TURN1_V54_NORM_CACHE[key] = s
        return s
    except Exception:
        return ""

def _turn1_v40_flatten_strings(obj, max_items=5000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)

    rec(obj)
    return " ".join(out)


def _turn1_v40_card_name(card):
    try:
        return card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_v40_is_opponent_state_dependent_access_card(card):
    """Return True for access cards whose draw/search amount depends on opponent state.

    Direct cached replacement for the original v40 guard. This intentionally
    focuses on access effects, not generic interaction with the opponent.
    Example blocked: Morty's Conviction, because its draw count depends on the
    opponent's Bench.
    Example not blocked: Iono-style text that affects both players but does not
    make our access count depend on the opponent's board size.
    """
    try:
        key = _turn1_v54_card_cache_key(card)
        cached = _TURN1_V54_OPP_DEP_ACCESS_CACHE.get(key)
        if cached is not None:
            return cached

        blob = _turn1_v40_norm(card)

        has_access = any(
            token in blob
            for token in [
                "draw",
                "search your deck",
                "search the deck",
                "look at the top",
                "reveal",
                "choose",
            ]
        )

        opponent_dependent = any(
            token in blob
            for token in [
                "for each of your opponent",
                "for each pokemon your opponent",
                "for each pokémon your opponent",
                "for each of your opponents",
                "opponent's benched pokemon",
                "opponent's benched pokémon",
                "opponents benched pokemon",
                "opponents benched pokémon",
                "your opponent's benched pokemon",
                "your opponent's benched pokémon",
                "your opponents benched pokemon",
                "your opponents benched pokémon",
                "number of cards in your opponent's hand",
                "number of cards in your opponents hand",
                "number of your opponent's benched",
                "number of your opponents benched",
            ]
        )

        result = bool(has_access and opponent_dependent)

        if len(_TURN1_V54_OPP_DEP_ACCESS_CACHE) < 250000:
            _TURN1_V54_OPP_DEP_ACCESS_CACHE[key] = result
        return result
    except Exception:
        return False


# TURN1_FIX_V54_MISSING_PLAY_CARD_ORIG_V55
# v54 replaced the helper above this block and accidentally removed
# the original v40 assignment. Restore the existing v40 play_card
# link so play_card can call the pre-v40 implementation.
_ORIG_PLAY_CARD_V40 = play_card

def play_card(*args, **kwargs):
    card = None

    # Common signature is play_card(st, card, ...). Keep it robust.
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, dict) and _turn1_v40_card_name(value):
            card = value
            break

    if card is not None and _turn1_v40_is_opponent_state_dependent_access_card(card):
        st = args[0] if args else kwargs.get("st")

        try:
            st.events.append(
                {
                    "event": "blocked_opponent_dependent_access_v40",
                    "card": _turn1_v40_card_name(card),
                    "reason": "Effect depends on opponent state that the Turn-1 simulator does not model.",
                }
            )
        except Exception:
            pass

        return False

    return _ORIG_PLAY_CARD_V40(*args, **kwargs)

# ---------------------------------------------------------------------
# TURN1_V73_STRICT_FILTER_SOURCE_CONJUNCTION
# ---------------------------------------------------------------------
# Root guard for search execution leaks.
#
# v70 made search execution source-aware, but there were still permissive
# fallbacks where a loose compiled/raw filter could allow a target even when the
# printed search text excluded it.  The observed failure was Buddy-Buddy Poffin
# selecting Basic Water Energy from a filter whose text was:
#   "up to 2 Basic Pokémon with 70 HP or less"
#
# This block is deliberately broad:
#   1. Interpret raw/compiled search filters strictly for Energy/Pokemon/Trainer
#      classes, HP caps, Bench/basic-Pokemon text, and typed Energy/Pokemon.
#   2. During source-bound search execution require BOTH:
#        - the source card can legally fetch the candidate, and
#        - the printed/compiled filter allows the candidate.
#
# So a card cannot search a target simply because one fallback says yes.
# The source card and the specific filter must agree.

_TURN1_V73_ORIG_FILTER_ALLOWS_CARD = filter_allows_card

def _turn1_v73_norm(value):
    import re, unicodedata
    s = str(value or '')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace('pokémon', 'pokemon').replace('’', "'").replace('`', "'")
    s = re.sub(r'[^a-z0-9{}\s\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _turn1_v73_blob(obj, depth=0):
    if obj is None or depth > 6:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        # Prefer text-bearing keys first, but include all values because compiled
        # filters vary by compiler generation.
        for key in (
            'raw_text', 'source_text', 'from_text', 'text', 'filter_text',
            'name', 'kind', 'type', 'subtype', 'supertype', 'target', 'targets',
            'selection', 'search_filter', 'filters', 'constraints', 'hp', 'max_hp',
        ):
            if key in obj:
                parts.append(str(key))
                parts.append(_turn1_v73_blob(obj.get(key), depth + 1))
        for k, v in obj.items():
            if k not in {
                'raw_text', 'source_text', 'from_text', 'text', 'filter_text',
                'name', 'kind', 'type', 'subtype', 'supertype', 'target', 'targets',
                'selection', 'search_filter', 'filters', 'constraints', 'hp', 'max_hp',
            }:
                parts.append(str(k))
                parts.append(_turn1_v73_blob(v, depth + 1))
        return ' '.join(p for p in parts if p)
    if isinstance(obj, (list, tuple, set)):
        return ' '.join(_turn1_v73_blob(x, depth + 1) for x in obj)
    return str(obj)

def _turn1_v73_card_blob(card):
    return _turn1_v73_norm(_turn1_v73_blob(card))

def _turn1_v73_name(card):
    try:
        return card_name(card)
    except Exception:
        if isinstance(card, dict):
            ident = card.get('identity') or {}
            return str(card.get('name') or card.get('card_name') or ident.get('name') or '')
        return ''

def _turn1_v73_supertype(card):
    if not isinstance(card, dict):
        return ''
    ident = card.get('identity') or {}
    return str(card.get('supertype') or ident.get('supertype') or '')

def _turn1_v73_subtypes(card):
    vals = []
    if isinstance(card, dict):
        ident = card.get('identity') or {}
        for src in (card, ident):
            if not isinstance(src, dict):
                continue
            for key in ('subtypes', 'subtype', 'trainerType'):
                v = src.get(key)
                if isinstance(v, (list, tuple, set)):
                    vals.extend(str(x) for x in v)
                elif v:
                    vals.append(str(v))
    return vals

def _turn1_v73_types(card):
    vals = []
    if isinstance(card, dict):
        ident = card.get('identity') or {}
        for src in (card, ident):
            if not isinstance(src, dict):
                continue
            v = src.get('types')
            if isinstance(v, (list, tuple, set)):
                vals.extend(str(x) for x in v)
            elif v:
                vals.append(str(v))
    return vals

def _turn1_v73_hp(card):
    if not isinstance(card, dict):
        return None
    ident = card.get('identity') or {}
    for src in (card, ident):
        if not isinstance(src, dict):
            continue
        hp = src.get('hp')
        try:
            return int(str(hp).strip())
        except Exception:
            pass
    return None

def _turn1_v73_is_pokemon(card):
    try:
        return bool(is_pokemon(card))
    except Exception:
        pass
    return 'pokemon' in _turn1_v73_norm(_turn1_v73_supertype(card))

def _turn1_v73_is_basic_pokemon(card):
    try:
        return bool(is_basic_pokemon(card))
    except Exception:
        pass
    return _turn1_v73_is_pokemon(card) and any('basic' in _turn1_v73_norm(x) for x in _turn1_v73_subtypes(card))

def _turn1_v73_is_energy(card):
    try:
        return bool(is_energy(card))
    except Exception:
        pass
    return 'energy' in _turn1_v73_norm(_turn1_v73_supertype(card) + ' ' + _turn1_v73_name(card))

def _turn1_v73_is_basic_energy(card):
    try:
        return bool(is_basic_energy(card))
    except Exception:
        pass
    blob = _turn1_v73_norm(_turn1_v73_supertype(card) + ' ' + _turn1_v73_name(card) + ' ' + ' '.join(_turn1_v73_subtypes(card)))
    return _turn1_v73_is_energy(card) and 'basic' in blob

def _turn1_v73_is_trainer(card):
    try:
        return bool(is_trainer(card))
    except Exception:
        pass
    return 'trainer' in _turn1_v73_norm(_turn1_v73_supertype(card))

def _turn1_v73_trainer_kind(card, kind):
    b = _turn1_v73_norm(_turn1_v73_supertype(card) + ' ' + _turn1_v73_name(card) + ' ' + ' '.join(_turn1_v73_subtypes(card)))
    k = _turn1_v73_norm(kind)
    if k == 'tool':
        return _turn1_v73_is_trainer(card) and ('tool' in b or 'pokemon tool' in b)
    return _turn1_v73_is_trainer(card) and k in b

_TURN1_V73_TYPES = {
    'grass': {'grass', 'g'},
    'fire': {'fire', 'r'},
    'water': {'water', 'w'},
    'lightning': {'lightning', 'electric', 'l'},
    'psychic': {'psychic', 'p'},
    'fighting': {'fighting', 'f'},
    'darkness': {'darkness', 'dark', 'd'},
    'metal': {'metal', 'steel', 'm'},
    'colorless': {'colorless', 'c'},
}

def _turn1_v73_card_has_type(card, typ):
    aliases = _TURN1_V73_TYPES.get(typ, {typ})
    vals = {_turn1_v73_norm(x) for x in _turn1_v73_types(card)}
    name = _turn1_v73_norm(_turn1_v73_name(card))
    return bool(vals.intersection(aliases)) or any((a + ' energy') in name for a in aliases)

def _turn1_v73_hp_ok(card, text):
    import re
    m = re.search(r'(\d+)\s*hp\s*or\s*less', text)
    if not m:
        return True
    hp = _turn1_v73_hp(card)
    return hp is not None and hp <= int(m.group(1))


# TURN1_V73_FROM_TEXT_BASIC_HP_FILTER_CANONICAL_V1
# Canonical filter matcher fix.
#
# Poffin-style compiled filters can arrive as:
#   {"from_text": "up to 2 Basic Pokémon with 70 HP or less"}
#
# This belongs in filter_allows_card/_turn1_v73_filter_decision, not in V78,
# because V78 correctly delegates to filter_allows_card before selecting.

def _turn1_v73_from_text_flatten_v1(obj, depth=0):
    if obj is None or depth > 6:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return " ".join(_turn1_v73_from_text_flatten_v1(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(str(k))
            parts.append(_turn1_v73_from_text_flatten_v1(v, depth + 1))
        return " ".join(parts)
    return str(obj)


def _turn1_v73_parse_int_v1(value):
    try:
        s = str(value or "").strip()
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return None


def _turn1_v73_find_hp_v1(obj, depth=0):
    if obj is None or depth > 6:
        return None

    if isinstance(obj, dict):
        for k, v in obj.items():
            if _turn1_v73_norm(k) == "hp":
                hp = _turn1_v73_parse_int_v1(v)
                if hp is not None:
                    return hp

        for v in obj.values():
            hp = _turn1_v73_find_hp_v1(v, depth + 1)
            if hp is not None:
                return hp

    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hp = _turn1_v73_find_hp_v1(v, depth + 1)
            if hp is not None:
                return hp

    return None


def _turn1_v73_card_supertype_norm_v1(card):
    if not isinstance(card, dict):
        return ""

    try:
        st = card_supertype(card)
        if st:
            return _turn1_v73_norm(st)
    except Exception:
        pass

    for container_key in ("identity", "gameplay", "printed", "raw_card"):
        container = card.get(container_key)
        if isinstance(container, dict):
            raw = container.get("supertype")
            if raw:
                return _turn1_v73_norm(raw)

    raw = card.get("supertype")
    if raw:
        return _turn1_v73_norm(raw)

    return ""


def _turn1_v73_card_is_basic_pokemon_v1(card):
    supertype = _turn1_v73_card_supertype_norm_v1(card)
    if supertype and supertype != "pokemon":
        return False

    try:
        if is_basic_pokemon(card):
            return True
    except Exception:
        pass

    blob = _turn1_v73_norm(_turn1_v73_from_text_flatten_v1(card))

    if supertype == "pokemon" and "basic" in blob:
        return True

    return False


def _turn1_v73_from_text_basic_hp_filter_decision_v1(filt, card):
    if not isinstance(filt, dict):
        return None

    raw_text = " ".join(
        str(filt.get(k) or "")
        for k in ("from_text", "source_text", "raw_text", "text", "filter_text", "description")
    )
    blob = _turn1_v73_norm(raw_text)

    if not blob:
        return None

    wants_basic_pokemon = "basic pokemon" in blob
    wants_hp_lte = "hp or less" in blob

    if not wants_basic_pokemon and not wants_hp_lte:
        return None

    if wants_basic_pokemon:
        if _turn1_v73_card_supertype_norm_v1(card) != "pokemon":
            return False
        if not _turn1_v73_card_is_basic_pokemon_v1(card):
            return False

    if wants_hp_lte:
        import re
        hp_limit = 70
        m = re.search(r"(\d{2,3})\s*hp\s*or\s*less", blob)
        if m:
            hp_limit = int(m.group(1))

        hp = _turn1_v73_find_hp_v1(card)

        # Fail closed if the card has no HP metadata. This prevents Poffin-like
        # text from accidentally accepting arbitrary Basic Pokémon with unknown HP.
        if hp is None:
            return False

        if hp > hp_limit:
            return False

    return True



def _turn1_v73_filter_decision(filt, card):
    # TURN1_V73_FROM_TEXT_BASIC_HP_FILTER_CANONICAL_V1
    from_text_basic_hp_decision = _turn1_v73_from_text_basic_hp_filter_decision_v1(filt, card)
    if from_text_basic_hp_decision is not None:
        return bool(from_text_basic_hp_decision)

    if not isinstance(filt, dict):
        return None
    text = _turn1_v73_norm(_turn1_v73_blob(filt))
    if not text:
        return None

    # Only make hard decisions for filters that clearly constrain a search/select.
    has_constraint_words = any(w in text for w in (
        'pokemon', 'energy', 'trainer', 'supporter', 'item', 'stadium', 'tool',
        'hp or less', 'bench', 'basic', 'ex', 'card'
    ))
    if not has_constraint_words:
        return None

    tests = []

    # Trainer/tool branches.  Handle Pokemon Tool before generic Pokemon.
    if 'pokemon tool' in text or ' tool' in (' ' + text):
        if 'tool' in text and 'pokemon' in text and 'hp or less' not in text and 'basic pokemon' not in text:
            tests.append(lambda c: _turn1_v73_trainer_kind(c, 'tool'))
    if 'supporter' in text:
        tests.append(lambda c: _turn1_v73_trainer_kind(c, 'supporter'))
    if 'stadium' in text:
        tests.append(lambda c: _turn1_v73_trainer_kind(c, 'stadium'))
    if 'item' in text and 'pokemon' not in text:
        tests.append(lambda c: _turn1_v73_trainer_kind(c, 'item'))
    if 'trainer' in text and not any(x in text for x in ('supporter', 'stadium', 'tool', 'item')):
        tests.append(lambda c: _turn1_v73_is_trainer(c))

    # Energy branches, including OR text such as Fighting Gong.
    for typ in _TURN1_V73_TYPES:
        aliases = _TURN1_V73_TYPES[typ]
        if any((a + ' energy') in text or ('{' + a + '} energy') in text for a in aliases):
            if 'basic' in text:
                tests.append(lambda c, typ=typ: _turn1_v73_is_basic_energy(c) and _turn1_v73_card_has_type(c, typ))
            else:
                tests.append(lambda c, typ=typ: _turn1_v73_is_energy(c) and _turn1_v73_card_has_type(c, typ))
    if 'basic energy' in text:
        tests.append(lambda c: _turn1_v73_is_basic_energy(c))
    elif 'energy' in text and not tests:
        tests.append(lambda c: _turn1_v73_is_energy(c))

    # Pokemon branches, including Bench/HP-limited searches.
    for typ in _TURN1_V73_TYPES:
        aliases = _TURN1_V73_TYPES[typ]
        if any((a + ' pokemon') in text or ('{' + a + '} pokemon') in text for a in aliases):
            if 'basic' in text or 'hp or less' in text or 'bench' in text:
                tests.append(lambda c, typ=typ, text=text: _turn1_v73_is_basic_pokemon(c) and _turn1_v73_card_has_type(c, typ) and _turn1_v73_hp_ok(c, text))
            else:
                tests.append(lambda c, typ=typ, text=text: _turn1_v73_is_pokemon(c) and _turn1_v73_card_has_type(c, typ) and _turn1_v73_hp_ok(c, text))
    if 'pokemon ex' in text:
        tests.append(lambda c: _turn1_v73_is_pokemon(c) and 'ex' in _turn1_v73_norm(_turn1_v73_name(c) + ' ' + ' '.join(_turn1_v73_subtypes(c))))
    if 'basic pokemon' in text or 'hp or less' in text or ('bench' in text and 'pokemon' in text):
        tests.append(lambda c, text=text: _turn1_v73_is_basic_pokemon(c) and _turn1_v73_hp_ok(c, text))
    elif 'pokemon' in text and 'pokemon tool' not in text:
        tests.append(lambda c, text=text: _turn1_v73_is_pokemon(c) and _turn1_v73_hp_ok(c, text))

    if tests:
        return any(test(card) for test in tests)

    # If it is clearly a search/select filter but we cannot classify the target,
    # do not let it become an any-card search.
    if any(x in text for x in ('search', 'reveal', 'choose', 'put into your hand', 'put onto your bench')):
        return False
    return None



# TURN1_STRICT_STRUCTURED_FILTER_V1
def _turn1_strict_filter_norm(value):
    try:
        return norm(value)
    except Exception:
        import re as _re
        s = str(value or "").lower().replace("pokémon", "pokemon").replace("’", "'")
        return _re.sub(r"[^a-z0-9{}'\s_-]+", " ", s).strip()


def _turn1_strict_filter_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None]
    return [value]


def _turn1_strict_card_subtypes(card):
    out = []
    try:
        out.extend(card_subtypes(card) or [])
    except Exception:
        pass

    if isinstance(card, dict):
        for obj in (card, card.get("identity"), card.get("gameplay"), card.get("raw_card")):
            if not isinstance(obj, dict):
                continue
            for key in ("subtype", "subtypes"):
                value = obj.get(key)
                if isinstance(value, (list, tuple, set)):
                    out.extend(value)
                elif value:
                    out.append(value)

    # Some records are incomplete but the printed name is still enough for ex.
    try:
        name = card_name(card)
    except Exception:
        name = card.get("name") if isinstance(card, dict) else ""
    if _turn1_strict_filter_norm(name).endswith(" ex"):
        out.append("ex")

    return {_turn1_strict_filter_norm(x) for x in out if str(x or "").strip()}


def _turn1_strict_card_types(card):
    out = []
    if isinstance(card, dict):
        for obj in (card, card.get("identity"), card.get("gameplay"), card.get("raw_card")):
            if not isinstance(obj, dict):
                continue
            value = obj.get("types") or obj.get("type")
            if isinstance(value, (list, tuple, set)):
                out.extend(value)
            elif value:
                out.append(value)
    return {_turn1_strict_filter_norm(x) for x in out if str(x or "").strip()}


def _turn1_strict_card_supertype(card):
    try:
        return _turn1_strict_filter_norm(card_supertype(card))
    except Exception:
        if isinstance(card, dict):
            for obj in (card, card.get("identity"), card.get("raw_card")):
                if isinstance(obj, dict) and obj.get("supertype"):
                    return _turn1_strict_filter_norm(obj.get("supertype"))
        return ""


def _turn1_strict_card_name(card):
    try:
        return _turn1_strict_filter_norm(card_name(card))
    except Exception:
        return _turn1_strict_filter_norm(card.get("name") if isinstance(card, dict) else "")


def _turn1_strict_card_has_rule_box(card):
    name = _turn1_strict_card_name(card)
    subtypes = _turn1_strict_card_subtypes(card)

    rule_box_markers = {
        "ex", "v", "vmax", "vstar", "gx", "ex", "break", "prism star",
        "radiant", "tera",
    }
    if any(x in subtypes for x in rule_box_markers):
        return True

    if (
        name.endswith(" ex")
        or name.endswith(" v")
        or name.endswith(" vmax")
        or name.endswith(" vstar")
        or name.endswith(" gx")
    ):
        return True

    if isinstance(card, dict):
        for obj in (card, card.get("identity"), card.get("gameplay"), card.get("raw_card")):
            if not isinstance(obj, dict):
                continue
            for key in ("rule_box", "has_rule_box", "is_rule_box"):
                if obj.get(key) is True:
                    return True
            rules = str(obj.get("rules") or "")
            if rules.strip():
                return True

    return False


def _turn1_strict_card_is_pokemon_ex(card):
    if _turn1_strict_card_supertype(card) != "pokemon":
        return False
    if "ex" in _turn1_strict_card_subtypes(card):
        return True
    return _turn1_strict_card_name(card).endswith(" ex")


def _turn1_strict_structured_filter_rejects(filt, card):
    """Return True only when an explicit structured filter constraint rejects card.

    This is intentionally a rejection layer, not a full replacement matcher.
    It prevents later raw-text/fallback logic from widening structured filters.
    """
    if not isinstance(filt, dict) or not filt:
        return False

    # supertype is conjunctive when explicitly present.
    if filt.get("supertype") is not None:
        required = _turn1_strict_filter_norm(filt.get("supertype"))
        actual = _turn1_strict_card_supertype(card)
        if required and actual != required:
            return True

    # subtype/subtypes are conjunctive constraints. Example:
    # {"supertype": "Pokémon", "subtype": "ex"} must reject Basic N's Zorua.
    required_subtypes = []
    required_subtypes.extend(_turn1_strict_filter_list(filt.get("subtype")))
    required_subtypes.extend(_turn1_strict_filter_list(filt.get("subtypes")))
    required_subtypes.extend(_turn1_strict_filter_list(filt.get("required_subtype")))
    required_subtypes.extend(_turn1_strict_filter_list(filt.get("required_subtypes")))

    if required_subtypes:
        actual_subtypes = _turn1_strict_card_subtypes(card)
        for req in required_subtypes:
            req_n = _turn1_strict_filter_norm(req)
            if req_n and req_n not in actual_subtypes:
                return True

    # types are also explicit constraints.
    required_types = []
    required_types.extend(_turn1_strict_filter_list(filt.get("type")))
    required_types.extend(_turn1_strict_filter_list(filt.get("types")))
    required_types.extend(_turn1_strict_filter_list(filt.get("energy_type")))
    required_types.extend(_turn1_strict_filter_list(filt.get("energy_types")))

    if required_types:
        actual_types = _turn1_strict_card_types(card)
        for req in required_types:
            req_n = _turn1_strict_filter_norm(req)
            if req_n and req_n not in actual_types:
                return True

    if filt.get("is_pokemon_ex") is True:
        if not _turn1_strict_card_is_pokemon_ex(card):
            return True

    for key in ("rule_box", "has_rule_box", "is_rule_box"):
        if filt.get(key) is True and not _turn1_strict_card_has_rule_box(card):
            return True
        if filt.get(key) is False and _turn1_strict_card_has_rule_box(card):
            return True

    for key in ("exclude_rule_box", "requires_no_rule_box"):
        if filt.get(key) is True and _turn1_strict_card_has_rule_box(card):
            return True

    return False

def filter_allows_card(filt, card):
    if _turn1_strict_structured_filter_rejects(filt, card):
        return False
    decision = _turn1_v73_filter_decision(filt, card)
    if decision is not None:
        return bool(decision)
    return _TURN1_V73_ORIG_FILTER_ALLOWS_CARD(filt, card)

# Strengthen v70 source-bound search execution: the source card and the exact
# compiled/raw filter must BOTH allow the candidate.  This prevents cases where
# card_directly_searches_target or a filter fallback alone is too broad.
try:
    _TURN1_V73_ORIG_SOURCE_CAN_SELECT_TARGET = _turn1_v70_source_can_select_target_card
except Exception:
    _TURN1_V73_ORIG_SOURCE_CAN_SELECT_TARGET = None

if _TURN1_V73_ORIG_SOURCE_CAN_SELECT_TARGET is not None:
    def _turn1_v70_source_can_select_target_card(st, filt, candidate, target_norm, step=None, stage=None, source_card=None):
        if not isinstance(candidate, dict) or not target_matches(candidate, target_norm):
            return False
        if not filter_allows_card(filt, candidate):
            try:
                st.log.append({
                    'event': 'blocked_v73_filter_candidate_mismatch',
                    'stage': stage,
                    'candidate': card_name(candidate),
                    'filter': filt,
                })
            except Exception:
                pass
            return False
        src = _turn1_v70_current_source_card(st, stage=stage, source_card=source_card)
        if isinstance(src, dict):
            ok = bool(_turn1_v70_source_can_fetch_candidate(src, candidate, getattr(st, 'deck', [])))
            if not ok:
                try:
                    st.log.append({
                        'event': 'blocked_v73_source_candidate_mismatch',
                        'stage': stage,
                        'source': card_name(src),
                        'candidate': card_name(candidate),
                        'filter': filt,
                    })
                except Exception:
                    pass
            return ok
        # Source context missing: allow only a few explicit legacy narrow paths.
        try:
            current_card_name = st.line[-1] if getattr(st, 'line', None) else ''
            if norm(current_card_name) == 'ultra ball':
                return card_supertype(candidate) == 'Pokémon'
        except Exception:
            pass
        return False

try:
    _TURN1_V73_ORIG_SOURCE_CAN_FETCH_ENABLER = _turn1_v70_source_can_fetch_enabler
except Exception:
    _TURN1_V73_ORIG_SOURCE_CAN_FETCH_ENABLER = None

if _TURN1_V73_ORIG_SOURCE_CAN_FETCH_ENABLER is not None:
    def _turn1_v70_source_can_fetch_enabler(st, filt, candidate, target_norm, going, source_card=None, source_step=None, stage=None):
        if not isinstance(candidate, dict):
            return False
        if not filter_allows_card(filt, candidate):
            return False
        if not card_can_be_played_from_hand(candidate, going, st.supporter_used):
            return False
        src = _turn1_v70_current_source_card(st, stage=stage, source_card=source_card)
        if isinstance(src, dict):
            if not _turn1_v70_source_can_fetch_candidate(src, candidate, getattr(st, 'deck', [])):
                return False
        else:
            return False
        direct = bool(card_directly_searches_target(candidate, target_norm, _turn1_v70_deck_with_candidate(getattr(st, 'deck', []), candidate)))
        draw_power = int(card_draw_power(candidate) or 0)
        try:
            play_score_no_chain = float(score_playable_card(candidate, st, target_norm, going, False))
        except Exception:
            play_score_no_chain = 0.0
        return direct or draw_power > 0 or play_score_no_chain > 0

# ---------------------------------------------------------------------
# TURN1_V74_SOURCE_TEXT_DIRECT_SEARCH_GUARD
# ---------------------------------------------------------------------
# Root fix for direct-search false positives such as:
#   Poké Pad -> Wally's Compassion
#
# Earlier guards correctly fixed many executor leaks, but
# card_directly_searches_target() could still return True when a compiled
# filter was vague (for example {"from_text": True}) and the source card text
# was not used to enforce the target class. This block makes direct-search
# reachability source-text aware for all search cards.
#
# Examples:
# - Poké Pad: Pokemon without a Rule Box only; never Supporters.
# - Buddy-Buddy Poffin: Basic Pokemon with HP cap only; never Energy.
# - Earthen Vessel / Shivery Chill: Basic typed Energy only.
# - Fighting Gong: Basic Fighting Energy OR Basic Fighting Pokemon.
# - Irida-style text: Water Pokemon OR Item.
# ---------------------------------------------------------------------

_TURN1_V74_ORIG_CARD_DIRECTLY_SEARCHES_TARGET = card_directly_searches_target


def _turn1_v74_norm(value):
    try:
        return norm(value).replace('pokémon', 'pokemon')
    except Exception:
        import re as _re, unicodedata as _unicodedata
        s = _unicodedata.normalize('NFKD', str(value or ''))
        s = ''.join(ch for ch in s if not _unicodedata.combining(ch))
        s = s.lower().replace('pokémon', 'pokemon')
        s = _re.sub(r'[^a-z0-9{}]+', ' ', s)
        return _re.sub(r'\s+', ' ', s).strip()


def _turn1_v74_blob(obj, depth=0):
    if obj is None or depth > 5:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        preferred = (
            'name', 'card_name', 'text', 'raw_text', 'source_text',
            'combined_text', 'rules', 'abilities_text', 'attacks_text',
            'effect_text', 'description', 'filter', 'card_filter',
            'selection', 'target', 'targets', 'from_text', 'constraints',
            'supertype', 'subtypes', 'types', 'hp', 'max_hp',
        )
        parts = []
        for k in preferred:
            if k in obj:
                parts.append(str(k))
                parts.append(_turn1_v74_blob(obj.get(k), depth + 1))
        for k, v in obj.items():
            if k not in preferred:
                parts.append(str(k))
                parts.append(_turn1_v74_blob(v, depth + 1))
        return ' '.join(p for p in parts if p)
    if isinstance(obj, (list, tuple, set)):
        return ' '.join(_turn1_v74_blob(x, depth + 1) for x in obj)
    return str(obj)


def _turn1_v74_card_source_blob(card, filt=None, step=None):
    parts = []
    if isinstance(step, dict):
        for k in ('text', 'raw_text', 'source_text', 'effect_text', 'description'):
            if step.get(k):
                parts.append(str(step.get(k)))
        try:
            parts.append(step_text(step))
        except Exception:
            pass
        parts.append(_turn1_v74_blob(step))
    if isinstance(filt, dict):
        try:
            parts.append(filter_text_blob(filt))
        except Exception:
            pass
        parts.append(_turn1_v74_blob(filt))
    if isinstance(card, dict):
        # Include full source card text only after step/filter text. This catches
        # generated filters like {"from_text": true}, where the useful search
        # restriction remains only on the source card object.
        parts.append(_turn1_v74_blob(card))
        try:
            for eff in iter_effects(card):
                parts.append(_turn1_v74_blob(eff))
        except Exception:
            pass
    return _turn1_v74_norm(' '.join(p for p in parts if p))


def _turn1_v74_card_name(card):
    try:
        return str(card_name(card))
    except Exception:
        if isinstance(card, dict):
            ident = card.get('identity') or {}
            return str(card.get('name') or card.get('card_name') or ident.get('name') or '')
        return ''


def _turn1_v74_supertype(card):
    try:
        return _turn1_v74_norm(card_supertype(card))
    except Exception:
        if isinstance(card, dict):
            ident = card.get('identity') or {}
            return _turn1_v74_norm(card.get('supertype') or ident.get('supertype') or '')
        return ''


def _turn1_v74_subtypes(card):
    try:
        return {_turn1_v74_norm(x) for x in card_subtypes(card)}
    except Exception:
        vals = []
        if isinstance(card, dict):
            ident = card.get('identity') or {}
            for src in (card, ident):
                if isinstance(src, dict):
                    v = src.get('subtypes') or src.get('subtype')
                    if isinstance(v, (list, tuple, set)):
                        vals.extend(v)
                    elif v:
                        vals.append(v)
        return {_turn1_v74_norm(x) for x in vals}


def _turn1_v74_types(card):
    try:
        return {_turn1_v74_norm(x) for x in card_types(card)}
    except Exception:
        vals = []
        if isinstance(card, dict):
            ident = card.get('identity') or {}
            for src in (card, ident):
                if isinstance(src, dict):
                    v = src.get('types')
                    if isinstance(v, (list, tuple, set)):
                        vals.extend(v)
                    elif v:
                        vals.append(v)
        return {_turn1_v74_norm(x) for x in vals}


def _turn1_v74_hp(card):
    vals = []
    if isinstance(card, dict):
        vals.extend([card.get('hp'), card.get('HP'), card.get('raw_hp')])
        for k in ('identity', 'gameplay', 'raw_card', 'source'):
            obj = card.get(k)
            if isinstance(obj, dict):
                vals.extend([obj.get('hp'), obj.get('HP'), obj.get('raw_hp')])
    import re as _re
    for v in vals:
        if v is None:
            continue
        m = _re.search(r'\d+', str(v))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    return None


def _turn1_v74_is_pokemon(card):
    try:
        return card_supertype(card) == 'Pokémon'
    except Exception:
        return _turn1_v74_supertype(card) in {'pokemon', 'pokémon'}


def _turn1_v74_is_basic_pokemon(card):
    try:
        return bool(is_basic_pokemon(card))
    except Exception:
        return _turn1_v74_is_pokemon(card) and 'basic' in _turn1_v74_subtypes(card)


def _turn1_v74_is_energy(card):
    try:
        return bool(is_energy(card))
    except Exception:
        return _turn1_v74_supertype(card) == 'energy'


def _turn1_v74_is_basic_energy(card):
    subs = _turn1_v74_subtypes(card)
    name = _turn1_v74_norm(_turn1_v74_card_name(card))
    return _turn1_v74_is_energy(card) and ('basic' in subs or name.startswith('basic '))


def _turn1_v74_is_trainer(card):
    try:
        return bool(is_trainer(card))
    except Exception:
        return _turn1_v74_supertype(card) == 'trainer'


def _turn1_v74_is_supporter(card):
    try:
        return bool(is_supporter(card))
    except Exception:
        return _turn1_v74_is_trainer(card) and 'supporter' in _turn1_v74_subtypes(card)


def _turn1_v74_trainer_kind(card, kind):
    k = _turn1_v74_norm(kind)
    b = _turn1_v74_norm(_turn1_v74_card_name(card) + ' ' + ' '.join(_turn1_v74_subtypes(card)))
    if k == 'item':
        return _turn1_v74_is_trainer(card) and 'item' in b
    if k == 'tool':
        return _turn1_v74_is_trainer(card) and 'tool' in b
    if k == 'stadium':
        return _turn1_v74_is_trainer(card) and 'stadium' in b
    if k == 'supporter':
        return _turn1_v74_is_supporter(card)
    return _turn1_v74_is_trainer(card)


_TURN1_V74_TYPES = {
    'grass': {'grass', 'g'},
    'fire': {'fire', 'r'},
    'water': {'water', 'w'},
    'lightning': {'lightning', 'electric', 'l'},
    'psychic': {'psychic', 'p'},
    'fighting': {'fighting', 'f'},
    'darkness': {'darkness', 'dark', 'd'},
    'metal': {'metal', 'steel', 'm'},
    'colorless': {'colorless', 'c'},
}


def _turn1_v74_card_has_type(card, typ):
    aliases = _TURN1_V74_TYPES.get(typ, {typ})
    vals = _turn1_v74_types(card)
    name = _turn1_v74_norm(_turn1_v74_card_name(card))
    return bool(vals.intersection(aliases)) or any((a + ' energy') in name for a in aliases)


def _turn1_v74_has_rule_box(card):
    b = _turn1_v74_norm(_turn1_v74_card_name(card) + ' ' + ' '.join(_turn1_v74_subtypes(card)) + ' ' + _turn1_v74_blob(card))
    if 'rule box' in b:
        return True
    # Practical rule-box families. This intentionally errs conservative for
    # turn-1 access legality.
    rule_terms = {'ex', 'gx', 'v', 'vmax', 'vstar', 'prism star', 'radiant'}
    subs = _turn1_v74_subtypes(card)
    if subs.intersection(rule_terms):
        return True
    name = _turn1_v74_norm(_turn1_v74_card_name(card))
    return any((' ' + t) in (' ' + name) for t in (' ex', ' gx', ' v', ' vmax', ' vstar'))


def _turn1_v74_hp_ok(card, text):
    import re as _re
    m = _re.search(r'(\d+)\s*hp\s*or\s*less', text)
    if not m:
        return True
    hp = _turn1_v74_hp(card)
    return hp is not None and hp <= int(m.group(1))


def _turn1_v74_source_text_allows_candidate(action_card, filt, target_card, step=None):
    text = _turn1_v74_card_source_blob(action_card, filt=filt, step=step)
    if not text:
        return True

    # If this does not look like a search/select effect, don't make a hard
    # source-text decision here.
    searchish = any(x in text for x in (
        'search your deck', 'search the deck', 'look at the top', 'reveal',
        'choose', 'put into your hand', 'put it into your hand',
        'put them into your hand', 'put onto your bench', 'put them onto your bench',
    ))
    if not searchish:
        return True

    tests = []

    # Trainer branches. Avoid treating Pokemon Tool as Pokemon.
    if 'pokemon tool' in text or 'pokémon tool' in text:
        tests.append(lambda c: _turn1_v74_trainer_kind(c, 'tool'))
    if 'supporter' in text:
        tests.append(lambda c: _turn1_v74_trainer_kind(c, 'supporter'))
    if 'stadium' in text:
        tests.append(lambda c: _turn1_v74_trainer_kind(c, 'stadium'))
    if 'item' in text and 'pokemon' not in text.replace('pokemon tool', ''):
        tests.append(lambda c: _turn1_v74_trainer_kind(c, 'item'))
    if 'trainer' in text and not any(x in text for x in ('supporter', 'stadium', 'item', 'tool')):
        tests.append(lambda c: _turn1_v74_is_trainer(c))

    # Typed Energy / Pokemon branches.
    for typ, aliases in _TURN1_V74_TYPES.items():
        if any((a + ' energy') in text or ('{' + a + '} energy') in text for a in aliases):
            if 'basic' in text:
                tests.append(lambda c, typ=typ: _turn1_v74_is_basic_energy(c) and _turn1_v74_card_has_type(c, typ))
            else:
                tests.append(lambda c, typ=typ: _turn1_v74_is_energy(c) and _turn1_v74_card_has_type(c, typ))
        if any((a + ' pokemon') in text or ('{' + a + '} pokemon') in text for a in aliases):
            if 'basic' in text or 'hp or less' in text or 'bench' in text:
                tests.append(lambda c, typ=typ, text=text: _turn1_v74_is_basic_pokemon(c) and _turn1_v74_card_has_type(c, typ) and _turn1_v74_hp_ok(c, text))
            else:
                tests.append(lambda c, typ=typ, text=text: _turn1_v74_is_pokemon(c) and _turn1_v74_card_has_type(c, typ) and _turn1_v74_hp_ok(c, text))

    # Untyped class branches.
    if 'basic energy' in text:
        tests.append(lambda c: _turn1_v74_is_basic_energy(c))
    elif ' energy' in (' ' + text) or text.startswith('energy'):
        tests.append(lambda c: _turn1_v74_is_energy(c))

    if 'pokemon ex' in text or 'pokémon ex' in text:
        tests.append(lambda c: _turn1_v74_is_pokemon(c) and 'ex' in _turn1_v74_subtypes(c))
    if 'basic pokemon' in text or 'basic pokémon' in text or 'hp or less' in text or ('bench' in text and 'pokemon' in text):
        tests.append(lambda c, text=text: _turn1_v74_is_basic_pokemon(c) and _turn1_v74_hp_ok(c, text))
    elif 'pokemon' in text and 'pokemon tool' not in text:
        tests.append(lambda c, text=text: _turn1_v74_is_pokemon(c) and _turn1_v74_hp_ok(c, text))

    # Rule Box exclusion. Poké Pad-style text is usually "doesn't have a Rule
    # Box". Apply it in addition to the Pokemon test above.
    excludes_rule_box = (
        'doesn t have a rule box' in text
        or "doesn't have a rule box" in text
        or 'does not have a rule box' in text
        or 'no rule box' in text
    )

    if tests:
        ok = any(test(target_card) for test in tests)
        if ok and excludes_rule_box and _turn1_v74_has_rule_box(target_card):
            return False
        return ok

    # Generic "search for a card" effects are allowed. Otherwise, if the text
    # is search-like but unclassified, keep the older behavior by returning True.
    return True


def card_directly_searches_target(card: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    # TURN1_V74_SOURCE_TEXT_DIRECT_SEARCH_GUARD
    target_cards = [c for c in deck if target_matches(c, target_norm)]
    if not target_cards:
        return False

    # Preserve explicit hard-coded cards, but still avoid Trainer targets for
    # Pokemon-only special searches through their own predicates.
    try:
        if card_specific_directly_searches_target(card, target_norm, deck):
            return True
    except Exception:
        pass

    for step in meaningful_steps(card):
        if step.get('op') not in {'search_deck', 'choose_cards', 'put_card_into_hand', 'put_card_onto_bench'}:
            continue
        filt = extract_filter(step)
        for tc in target_cards:
            try:
                filter_ok = bool(filter_allows_card(filt, tc))
            except Exception:
                filter_ok = True
            if not filter_ok:
                continue
            if _turn1_v74_source_text_allows_candidate(card, filt, tc, step=step):
                return True

    return False

# ---------------------------------------------------------------------
# TURN1_V78_EXECUTE_STEPS_SOURCE_TEXT_GUARD
# ---------------------------------------------------------------------
# Canonical runtime guard for execute_steps(search_deck).
#
# The remaining leak after v74/v75/v76 was not card_directly_searches_target
# and not the goal-aware selector.  The generic execute_steps(search_deck)
# branch could still accept a candidate through a vague compiled filter such as
# {"from_text": True}.  That allowed logs like:
#     source/stage: after_play_Poké Pad
#     selected: Wally's Compassion
# even though Poké Pad's own source_text says it searches a Pokémon without a
# Rule Box.
#
# This guard is called directly by execute_steps before remove_first_matching can
# remove a card from deck.  A candidate is selectable only when:
#   1. it matches the requested target_norm,
#   2. the compiled filter allows it,
#   3. source_text on the exact step allows it, and
#   4. if a source card can be resolved, that source card can directly fetch it.
#
# The important detail is step-level source_text.  It remains available even
# when source-card inference is stale or points at a previous action.

def _turn1_v78_filter_from_text_only(filt):
    return isinstance(filt, dict) and bool(filt.get('from_text')) and len(filt) == 1


def _turn1_v78_step_has_source_text(step):
    if not isinstance(step, dict):
        return False
    for key in ('source_text', 'text', 'raw_text', 'effect_text', 'description'):
        try:
            if step.get(key):
                return True
        except Exception:
            pass
    return False


def _turn1_v78_runtime_can_select_search_target(st, filt, candidate, target_norm, step=None, stage=None, source_card=None):
    # TURN1_V78_EXECUTE_STEPS_SOURCE_TEXT_GUARD
    if not isinstance(candidate, dict):
        return False

    try:
        if not target_matches(candidate, target_norm):
            return False
    except Exception:
        return False

    try:
        if not filter_allows_card(filt, candidate):
            try:
                st.log.append({
                    'event': 'blocked_v78_filter_candidate_mismatch',
                    'stage': stage,
                    'candidate': card_name(candidate),
                    'filter': filt,
                })
            except Exception:
                pass
            return False
    except Exception:
        return False

    # Step-level source text is the most reliable guard for semantic-known
    # effects like Poké Pad, where the compiled filter may only say
    # {"from_text": True} but the step carries the real printed source_text.
    step_text_decision = None
    if '_turn1_v74_source_text_allows_candidate' in globals():
        try:
            step_text_decision = bool(_turn1_v74_source_text_allows_candidate(None, filt, candidate, step=step))
        except Exception:
            step_text_decision = None

    if step_text_decision is False:
        try:
            st.log.append({
                'event': 'blocked_v78_step_source_text_candidate_mismatch',
                'stage': stage,
                'candidate': card_name(candidate),
                'filter': filt,
                'source_text': (step or {}).get('source_text') if isinstance(step, dict) else None,
            })
        except Exception:
            pass
        return False

    src = None
    try:
        src = _turn1_v70_current_source_card(st, stage=stage, source_card=source_card)
    except Exception:
        src = source_card if isinstance(source_card, dict) else None

    if isinstance(src, dict):
        try:
            ok = bool(_turn1_v70_source_can_fetch_candidate(src, candidate, getattr(st, 'deck', [])))
        except Exception:
            ok = False
        if not ok:
            try:
                st.log.append({
                    'event': 'blocked_v78_source_candidate_mismatch',
                    'stage': stage,
                    'source': card_name(src),
                    'candidate': card_name(candidate),
                    'filter': filt,
                })
            except Exception:
                pass
        return ok

    # No source card resolved.  If the step has explicit source_text, the v74
    # source-text decision is enough.  This preserves legal compiled effects
    # while blocking vague from_text-only effects with no source context.
    if _turn1_v78_step_has_source_text(step):
        return bool(step_text_decision)

    # With a vague from_text-only filter and no source card/text, fail closed:
    # this is exactly the category that produced illegal cross-type searches.
    if _turn1_v78_filter_from_text_only(filt):
        try:
            st.log.append({
                'event': 'blocked_v78_missing_source_for_from_text_filter',
                'stage': stage,
                'candidate': card_name(candidate),
                'filter': filt,
            })
        except Exception:
            pass
        return False

    return True

