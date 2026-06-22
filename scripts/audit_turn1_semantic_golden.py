from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any


SEMANTICS = Path("data/compiled_cards/auto/compiled_cards_all.turn1_semantics.json")


def norm(value: Any) -> str:
    s = str(value or "")
    s = (
        s.replace("Pokémon", "Pokemon")
        .replace("PokΘmon", "Pokemon")
        .replace("é", "e")
        .replace("É", "E")
        .replace("’", "'")
    )
    s = s.lower()
    s = re.sub(r"[^a-z0-9']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_cards(data: Any):
    if isinstance(data, list):
        yield from (x for x in data if isinstance(x, dict))
    elif isinstance(data, dict):
        for key in ["cards", "data", "compiled_cards", "items", "results"]:
            if isinstance(data.get(key), list):
                yield from (x for x in data[key] if isinstance(x, dict))
                return
        yield from (v for v in data.values() if isinstance(v, dict))


def card_name(card: dict[str, Any]) -> str:
    for key in ["name", "card_name", "cardName"]:
        if card.get(key):
            return str(card[key])
    ident = card.get("identity") or {}
    if isinstance(ident, dict):
        return str(ident.get("name") or "")
    return ""


def flat(obj: Any) -> str:
    out: list[str] = []

    def rec(v: Any) -> None:
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


def steps(obj: Any):
    if isinstance(obj, dict):
        if obj.get("op"):
            yield obj
        for v in obj.values():
            yield from steps(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from steps(v)


def find_cards(data: Any, name: str) -> list[dict[str, Any]]:
    n = norm(name)
    return [c for c in iter_cards(data) if norm(card_name(c)) == n]


def filt_blob(step: dict[str, Any]) -> str:
    return norm(json.dumps(step.get("filter") or step.get("selection_filter") or {}, ensure_ascii=False))


def step_blob(step: dict[str, Any]) -> str:
    return norm(json.dumps(step, ensure_ascii=False))


def _contains_key_value(obj: Any, keys: set[str], expected: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v is expected:
                return True
            if _contains_key_value(v, keys, expected):
                return True
    elif isinstance(obj, list):
        return any(_contains_key_value(v, keys, expected) for v in obj)
    return False


def _contains_truthy_key(obj: Any, keys: set[str]) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and bool(v):
                return True
            if _contains_truthy_key(v, keys):
                return True
    elif isinstance(obj, list):
        return any(_contains_truthy_key(v, keys) for v in obj)
    return False


def has_no_rule_box_filter(step: dict[str, Any]) -> bool:
    filt = step.get("filter") or step.get("selection_filter") or {}
    return (
        step.get("op") == "search_deck"
        and "pokemon" in filt_blob(step)
        and (
            _contains_key_value(filt, {"rule_box", "has_rule_box"}, False)
            or _contains_truthy_key(filt, {"exclude_rule_box", "requires_no_rule_box"})
            or "doesn t have a rule box" in step_blob(step)
            or "doesn't have a rule box" in step_blob(step)
            or "without a rule box" in step_blob(step)
        )
    )


def is_attract_customers_semantic_step(step: dict[str, Any]) -> bool:
    blob = step_blob(step)
    filt = step.get("filter") or step.get("selection_filter") or {}
    return (
        (
            "attract customers" in blob
            or "look at the top 6 cards" in blob
            or "top 6 cards of your deck" in blob
        )
        and (
            step.get("requires_active_spot") is True
            or "active spot" in blob
        )
        and (
            "supporter" in blob
            or "supporter" in filt_blob(step)
            or _contains_truthy_key(filt, {"supporter"})
        )
        and (
            step.get("destination") in {None, "self.hand"}
            or "self hand" in blob
        )
    )


def is_fighting_gong_safe_search_step(step: dict[str, Any]) -> bool:
    if step.get("op") != "search_deck":
        return False

    blob = step_blob(step)
    filt = filt_blob(step)

    mentions_fighting_target = (
        "basic fighting energy" in blob
        or "basic fighting pokemon" in blob
        or "basic fighting energy" in filt
        or "basic fighting pokemon" in filt
        or "fighting energy" in filt
        or "fighting pokemon" in filt
    )

    contaminated = (
        "pokemon ex" in filt
        or "is pokemon ex" in filt
        or "subtype ex" in filt
        or "rule box true" in filt
        or "rule_box true" in filt
        or "has rule box true" in filt
        or "has_rule_box true" in filt
    )

    return mentions_fighting_target and not contaminated


def is_fighting_gong_contaminated_search_step(step: dict[str, Any]) -> bool:
    if step.get("op") != "search_deck":
        return False

    blob = step_blob(step)
    filt = filt_blob(step)

    # Only judge steps that are actually Fighting Gong / Basic Fighting searches.
    # This avoids failing because some unrelated nested/global rule text mentions ex.
    is_fighting_gongish = (
        "fighting gong" in blob
        or "basic fighting energy" in blob
        or "basic fighting pokemon" in blob
        or "basic fighting energy" in filt
        or "basic fighting pokemon" in filt
        or "fighting energy" in filt
        or "fighting pokemon" in filt
    )

    contaminated = (
        "pokemon ex" in filt
        or "is pokemon ex" in filt
        or "subtype ex" in filt
        or "rule box true" in filt
        or "rule_box true" in filt
        or "has rule box true" in filt
        or "has_rule_box true" in filt
    )

    return is_fighting_gongish and contaminated


def amount_value(amount: Any) -> int | None:
    if isinstance(amount, int):
        return amount
    if isinstance(amount, dict):
        val = amount.get("value")
        if isinstance(val, int):
            return val
    return None


def any_step(cards: list[dict[str, Any]], pred) -> bool:
    return any(pred(s) for c in cards for s in steps(c))


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"PASS: {msg}")


def main() -> None:
    data = load_json(SEMANTICS)

    poke_pad = find_cards(data, "Poké Pad") or find_cards(data, "Poke Pad")
    ultra_ball = find_cards(data, "Ultra Ball")
    secret_box = find_cards(data, "Secret Box")
    cyrano = find_cards(data, "Cyrano")
    meowth = find_cards(data, "Meowth ex")
    tatsugiri = find_cards(data, "Tatsugiri")
    lillie = find_cards(data, "Lillie's Determination")
    zoroark = find_cards(data, "N's Zoroark ex")
    zorua = find_cards(data, "N's Zorua")
    fighting_gong = find_cards(data, "Fighting Gong")

    for name, cards in [
        ("Poké Pad", poke_pad),
        ("Ultra Ball", ultra_ball),
        ("Secret Box", secret_box),
        ("Cyrano", cyrano),
        ("Meowth ex", meowth),
        ("Tatsugiri", tatsugiri),
        ("Lillie's Determination", lillie),
        ("N's Zoroark ex", zoroark),
        ("N's Zorua", zorua),
        ("Fighting Gong", fighting_gong),
    ]:
        assert_true(bool(cards), f"found compiled card: {name}")

    assert_true(
        any_step(poke_pad, has_no_rule_box_filter),
        "Poké Pad searches Pokémon without Rule Box",
    )

    assert_true(
        any_step(
            cyrano,
            lambda s: s.get("op") == "search_deck"
            and "pokemon" in filt_blob(s)
            and "ex" in filt_blob(s)
            and (amount_value(s.get("amount")) == 3),
        ),
        "Cyrano searches up to 3 Pokémon ex",
    )

    assert_true(
        any_step(
            ultra_ball,
            lambda s: s.get("op") in {"discard_cards", "discard_card"}
            and amount_value(s.get("amount")) == 2
            and (s.get("required_to_play") is True or s.get("cost") is True),
        ),
        "Ultra Ball has required discard-2 cost",
    )

    assert_true(
        any_step(
            ultra_ball,
            lambda s: s.get("op") == "search_deck" and "pokemon" in filt_blob(s),
        ),
        "Ultra Ball searches Pokémon",
    )

    assert_true(
        any_step(
            secret_box,
            lambda s: s.get("op") in {"discard_cards", "discard_card"}
            and amount_value(s.get("amount")) == 3
            and (s.get("required_to_play") is True or s.get("cost") is True),
        ),
        "Secret Box has required discard-3 cost",
    )

    for subtype in ["item", "pokemon tool", "supporter", "stadium"]:
        assert_true(
            any_step(
                secret_box,
                lambda s, subtype=subtype: s.get("op") == "search_deck"
                and "trainer" in filt_blob(s)
                and subtype in filt_blob(s),
            ),
            f"Secret Box searches Trainer subtype: {subtype}",
        )

    assert_true(
        any_step(
            meowth,
            lambda s: s.get("op") == "search_deck"
            and s.get("destination") == "self.hand"
            and "supporter" in filt_blob(s),
        ),
        "Meowth ex Last-Ditch Catch searches Supporter to hand",
    )

    assert_true(
        not any_step(
            meowth,
            lambda s: "attract customers" in step_blob(s),
        ),
        "Meowth ex does not contain Attract Customers",
    )

    assert_true(
        any_step(tatsugiri, is_attract_customers_semantic_step),
        "Tatsugiri Attract Customers requires Active and selects Supporter",
    )

    assert_true(
        not any_step(
            tatsugiri,
            lambda s: s.get("op") in {"declare_attack", "deal_attack_damage"}
            and (
                "supporter" in filt_blob(s)
                or "supporter" in step_blob(s)
            )
            and (
                "top 6" in step_blob(s)
                or "look at" in step_blob(s)
                or s.get("look_at") is not None
                or s.get("selection_filter") is not None
            ),
        ),
        "Tatsugiri attack steps are not tagged as Attract Customers",
    )

    assert_true(
        any_step(fighting_gong, is_fighting_gong_safe_search_step),
        "Fighting Gong has a safe Basic Fighting search step",
    )

    assert_true(
        not any_step(fighting_gong, is_fighting_gong_contaminated_search_step),
        "Fighting Gong search filter is not contaminated by Pokémon ex semantics",
    )

    assert_true(
        any_step(
            lillie,
            lambda s: s.get("op") == "draw_cards" and s.get("random") is True,
        ),
        "Lillie's Determination is stochastic draw, not deterministic search",
    )

    print()
    print("All turn-1 semantic golden assertions passed.")


if __name__ == "__main__":
    main()
