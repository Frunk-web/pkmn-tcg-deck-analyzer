from __future__ import annotations

import copy
import gzip
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_IN = Path("data/compiled_cards/auto/compiled_cards_all.turn1_semantics.json")
DEFAULT_GZ = Path("data/compiled_cards/auto/compiled_cards_all.turn1_semantics.json.gz")


def norm_text(value: Any) -> str:
    s = str(value or "")
    s = (
        s.replace("Pokémon", "Pokemon")
        .replace("PokΘmon", "Pokemon")
        .replace("pokémon", "pokemon")
        .replace("pokθmon", "pokemon")
        .replace("é", "e")
        .replace("É", "E")
        .replace("’", "'")
        .replace("`", "'")
    )
    s = s.lower()
    s = re.sub(r"[^a-z0-9']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def flat_strings(obj: Any, max_items: int = 4000) -> str:
    out: list[str] = []

    def rec(v: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for vv in v.values():
                rec(vv)
        elif isinstance(v, list):
            for vv in v:
                rec(vv)

    rec(obj)
    return " ".join(out)


def source_blob(step: dict[str, Any], card_blob: str = "") -> str:
    pieces = [
        step.get("source_text"),
        step.get("text"),
        step.get("raw_text"),
        step.get("description"),
        step.get("effect_text"),
        card_blob,
    ]
    return norm_text(" ".join(str(x or "") for x in pieces))


def local_step_blob(step: dict[str, Any]) -> str:
    """Text local to this step/effect only; does not include the whole card blob."""
    pieces = [
        step.get("source_text"),
        step.get("text"),
        step.get("raw_text"),
        step.get("description"),
        step.get("effect_text"),
        step.get("ability_name"),
        step.get("attack_name"),
    ]
    return norm_text(" ".join(str(x or "") for x in pieces))


def is_attract_customers_local_text(text: str) -> bool:
    return (
        "attract customers" in text
        or ("look at the top 6 cards" in text and "supporter card" in text)
        or ("top 6 cards of your deck" in text and "supporter" in text and "active spot" in text)
    )


def amount_exact(n: int) -> dict[str, Any]:
    return {"mode": "exact", "value": int(n)}


def amount_up_to(n: int) -> dict[str, Any]:
    return {"mode": "up_to", "value": int(n)}


def is_search_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("op") == "search_deck"


def is_discard_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("op") in {"discard_cards", "discard_card"}


def make_search_step(
    *,
    source_step: dict[str, Any],
    source_text: str,
    filt: dict[str, Any],
    amount: Any = None,
    destination: str = "self.hand",
    reveal: bool = True,
    target_id: str = "searched_cards",
) -> dict[str, Any]:
    step = copy.deepcopy(source_step)
    step.update(
        {
            "op": "search_deck",
            "player": "self",
            "destination": destination,
            "filter": filt,
            "amount": amount if amount is not None else amount_exact(1),
            "reveal": reveal,
            "random": False,
            "choice_behavior": "player_selects_valid_cards_from_current_deck",
            "target_id": target_id,
            "source_text": source_text or source_step.get("source_text") or "",
            "normalized_by": "turn1_semantic_normalizer",
        }
    )
    return step


def normalize_discard_step(step: dict[str, Any], blob: str) -> dict[str, Any]:
    text = source_blob(step, blob)
    out = copy.deepcopy(step)

    m = re.search(r"discard (\d+) other cards?", text)
    if not m:
        m = re.search(r"discard (\d+) cards?", text)

    if m:
        out["amount"] = int(m.group(1))
        out["source"] = out.get("source") or "self.hand"
        out["source_zone"] = out.get("source_zone") or "hand"
        out["required_to_play"] = True
        out["cost"] = True
        out["normalized_by"] = "turn1_semantic_normalizer"

    return out


def normalize_single_step(step: dict[str, Any], blob: str) -> dict[str, Any]:
    out = copy.deepcopy(step)

    # Local effect text is used for effect-specific rewrites. The whole-card blob
    # is only a fallback for conservative metadata, because using it for search
    # filters can smear text from unrelated effects/rules onto this step.
    local_text = local_step_blob(out)
    text = local_text or source_blob(out, "")

    if is_discard_step(out):
        out = normalize_discard_step(out, local_text or blob)
        local_text = local_step_blob(out)
        text = local_text or source_blob(out, "")

    if not is_search_step(out):
        # Attract Customers is not deck search; it is top-6 Supporter selection.
        # IMPORTANT: use only local step/effect text here. Do not use the whole
        # card blob, or the metadata smears onto unrelated attacks such as Surf.
        if is_attract_customers_local_text(local_text):
            out["requires_active_spot"] = True
            out["selection_filter"] = {"supertype": "Trainer", "subtype": "Supporter"}
            out["look_at"] = {"zone": "self.deck", "amount": 6}
            out["destination"] = "self.hand"
            out["random"] = False
            out["ability_name"] = out.get("ability_name") or "Attract Customers"
            out["runtime_note"] = "top_6_select_supporter_not_deterministic_search"
            out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    # Poké Pad: Pokémon without Rule Box.
    if "doesn t have a rule box" in text or "doesn't have a rule box" in text or "without a rule box" in text:
        out["amount"] = amount_exact(1)
        out["destination"] = "self.hand"
        out["filter"] = {
            "supertype": "Pokémon",
            "rule_box": False,
            "has_rule_box": False,
            "exclude_rule_box": True,
            "requires_no_rule_box": True,
        }
        out["reveal"] = True
        out["random"] = False
        out["choice_behavior"] = "player_selects_valid_cards_from_current_deck"
        out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    # Cyrano-like: Pokémon ex.
    if "pokemon ex" in text and "supporter card" not in text:
        out["destination"] = "self.hand"
        out["filter"] = {
            "supertype": "Pokémon",
            "subtype": "ex",
            "is_pokemon_ex": True,
            "rule_box": True,
        }
        if "up to 3" in text:
            out["amount"] = amount_up_to(3)
        else:
            out["amount"] = out.get("amount") or amount_exact(1)
        out["reveal"] = True
        out["random"] = False
        out["choice_behavior"] = "player_selects_valid_cards_from_current_deck"
        out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    # Meowth ex / Last-Ditch Catch: Supporter to hand.
    if "last ditch" in text or (
        "when you play this pokemon from your hand onto your bench" in text
        and "supporter card" in text
        and "put it into your hand" in text
    ):
        out["amount"] = amount_exact(1)
        out["destination"] = "self.hand"
        out["filter"] = {"supertype": "Trainer", "subtype": "Supporter"}
        out["trigger"] = "when_played_from_hand_to_bench"
        out["usage_limit"] = {
            "scope": "per_turn_by_name_prefix",
            "prefix": "Last-Ditch",
            "max": 1,
        }
        out["reveal"] = True
        out["random"] = False
        out["choice_behavior"] = "player_selects_valid_cards_from_current_deck"
        out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    # Basic Water Energy / typed Basic Energy.
    m = re.search(r"up to (\d+) basic ([a-z]+) energy cards?", text)
    if m:
        energy_type = m.group(2).capitalize()
        out["amount"] = amount_up_to(int(m.group(1)))
        out["filter"] = {
            "supertype": "Energy",
            "subtype": "Basic",
            "energy_type": energy_type,
            "type": energy_type,
        }
        out["random"] = False
        out["choice_behavior"] = "player_selects_valid_cards_from_current_deck"
        out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    # Generic Pokémon search: normalize from raw_text/source_text.
    if "search your deck for a pokemon" in text or "search your deck for a pokemon reveal" in text:
        out["amount"] = out.get("amount") if out.get("amount") not in (None, {"mode": "from_text"}) else amount_exact(1)
        out["destination"] = "self.hand"
        out["filter"] = {"supertype": "Pokémon"}
        out["reveal"] = True
        out["random"] = False
        out["choice_behavior"] = "player_selects_valid_cards_from_current_deck"
        out["normalized_by"] = "turn1_semantic_normalizer"
        return out

    return out


def secret_box_replacement_steps(source_step: dict[str, Any], text: str) -> list[dict[str, Any]]:
    source_text = source_step.get("source_text") or text
    return [
        make_search_step(
            source_step=source_step,
            source_text=source_text,
            filt={"supertype": "Trainer", "subtype": "Item"},
            amount=amount_exact(1),
        ),
        make_search_step(
            source_step=source_step,
            source_text=source_text,
            filt={"supertype": "Trainer", "subtype": "Pokémon Tool"},
            amount=amount_exact(1),
        ),
        make_search_step(
            source_step=source_step,
            source_text=source_text,
            filt={"supertype": "Trainer", "subtype": "Supporter"},
            amount=amount_exact(1),
        ),
        make_search_step(
            source_step=source_step,
            source_text=source_text,
            filt={"supertype": "Trainer", "subtype": "Stadium"},
            amount=amount_exact(1),
        ),
    ]


def normalize_step_list(steps: list[Any], blob: str) -> list[Any]:
    out: list[Any] = []
    secret_box_inserted = False

    for step in steps:
        if isinstance(step, dict):
            text = local_step_blob(step) or source_blob(step, "")

            # Secret Box: split one broad text search into four typed searches.
            # Use only local step/effect text so another card/effect on the same
            # record cannot contaminate this search step.
            is_secret_box_search = (
                is_search_step(step)
                and "item card" in text
                and "pokemon tool card" in text
                and "supporter card" in text
                and "stadium card" in text
            )
            if is_secret_box_search:
                if not secret_box_inserted:
                    out.extend(secret_box_replacement_steps(step, step.get("source_text") or ""))
                    secret_box_inserted = True
                continue

            out.append(normalize_node(step, blob))
        elif isinstance(step, list):
            out.append(normalize_step_list(step, blob))
        else:
            out.append(step)

    return out


def normalize_node(obj: Any, blob: str) -> Any:
    if isinstance(obj, list):
        return normalize_step_list(obj, blob)

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in {"steps", "then", "else", "if_true", "if_false", "heads", "tails", "yes", "no", "branches"} and isinstance(v, list):
                out[k] = normalize_step_list(v, blob)
            else:
                out[k] = normalize_node(v, blob)

        # If the dict itself is a step, normalize after children.
        if isinstance(out.get("op"), str):
            out = normalize_single_step(out, blob)

        return out

    return obj


def normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    blob = flat_strings(card)
    return normalize_node(card, blob)


def iter_cards(container: Any) -> list[dict[str, Any]]:
    if isinstance(container, list):
        return [c for c in container if isinstance(c, dict)]
    if isinstance(container, dict):
        for key in ["cards", "data", "compiled_cards", "items", "results"]:
            if isinstance(container.get(key), list):
                return [c for c in container[key] if isinstance(c, dict)]
        return [v for v in container.values() if isinstance(v, dict)]
    return []


def normalize_container(container: Any) -> Any:
    if isinstance(container, list):
        return [normalize_card(c) if isinstance(c, dict) else c for c in container]

    if isinstance(container, dict):
        out = copy.deepcopy(container)
        for key in ["cards", "data", "compiled_cards", "items", "results"]:
            if isinstance(out.get(key), list):
                out[key] = [normalize_card(c) if isinstance(c, dict) else c for c in out[key]]
                return out

        for k, v in list(out.items()):
            if isinstance(v, dict):
                out[k] = normalize_card(v)
        return out

    return container


def main() -> None:
    in_path = DEFAULT_IN
    gz_path = DEFAULT_GZ

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    backup_path = in_path.with_suffix(in_path.suffix + ".bak_before_turn1_semantic_normalizer")
    backup_gz_path = gz_path.with_suffix(gz_path.suffix + ".bak_before_turn1_semantic_normalizer")

    if not backup_path.exists():
        shutil.copy2(in_path, backup_path)
    if gz_path.exists() and not backup_gz_path.exists():
        shutil.copy2(gz_path, backup_gz_path)

    raw = in_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    before_count = len(iter_cards(data))
    normalized = normalize_container(data)
    after_count = len(iter_cards(normalized))

    tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(in_path)

    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, separators=(",", ":"))

    print("Turn-1 semantic normalization complete.")
    print(f"Input/output JSON: {in_path}")
    print(f"Output gzip:        {gz_path}")
    print(f"Backup JSON:        {backup_path}")
    print(f"Backup gzip:        {backup_gz_path if gz_path.exists() else '(gzip did not exist before)'}")
    print(f"Card records:       {before_count} -> {after_count}")


if __name__ == "__main__":
    main()
