import csv
from pathlib import Path

raw_path = Path("data/decks/mega_lucario_raw_ptcgl.txt")
cards_csv = Path("data/all_cards.csv")
out_path = Path("data/decks/mega_lucario_exact_user_v01.txt")

TYPE_MAP = {
    "{G}": "Grass",
    "{R}": "Fire",
    "{W}": "Water",
    "{L}": "Lightning",
    "{P}": "Psychic",
    "{F}": "Fighting",
    "{D}": "Darkness",
    "{M}": "Metal",
    "{Y}": "Fairy",
    "{C}": "Colorless",
}

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

def clean_name(name: str) -> str:
    name = name.strip()
    for symbol, word in TYPE_MAP.items():
        name = name.replace(symbol, word)
    return " ".join(name.split())

def norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

def norm_num(s: str) -> str:
    x = norm(s)
    if x.isdigit():
        return str(int(x))
    return x.lstrip("0") or x

def get_col(row, *names):
    for name in names:
        if name in row and row[name]:
            return row[name]
    return ""

rows = []
with cards_csv.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

by_name_num = {}
by_name = {}

for row in rows:
    card_id = get_col(row, "card_id", "id")
    name = clean_name(get_col(row, "name"))
    number = get_col(row, "number", "collector_number")

    if not card_id or not name:
        continue

    by_name.setdefault(norm(name), []).append(row)

    if number:
        by_name_num.setdefault((norm(name), norm_num(number)), []).append(row)

lines_out = []
resolved_details = []
unresolved = []

for raw_line in raw_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()

    if not line or line.endswith(":"):
        continue

    parts = line.split()
    if len(parts) < 4 or not parts[0].isdigit():
        unresolved.append((line, "could_not_parse"))
        continue

    qty = parts[0]
    number = parts[-1]
    set_code = parts[-2]
    card_name = clean_name(" ".join(parts[1:-2]))

    candidates = by_name_num.get((norm(card_name), norm_num(number)), [])

    aliases = SET_ALIASES.get(set_code.upper(), [set_code])
    aliases_norm = [norm(a) for a in aliases]

    preferred = []
    for row in candidates:
        haystack = " ".join(
            str(get_col(row, col))
            for col in ["card_id", "id", "set_id", "set_code", "set_name", "series"]
        )
        hay_norm = norm(haystack)
        if any(alias in hay_norm for alias in aliases_norm):
            preferred.append(row)

    chosen = preferred or candidates

    if not chosen:
        name_candidates = by_name.get(norm(card_name), [])
        chosen = name_candidates

    if not chosen:
        unresolved.append((line, "no_match"))
        continue

    card_id = get_col(chosen[0], "card_id", "id")
    chosen_name = clean_name(get_col(chosen[0], "name"))
    chosen_num = get_col(chosen[0], "number", "collector_number")

    lines_out.append(f"{qty} {card_id}")
    resolved_details.append((line, f"{qty} {card_id}", chosen_name, chosen_num))

out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

print("wrote:", out_path)
print("resolved lines:", len(lines_out))
print("deck count:", sum(int(x.split()[0]) for x in lines_out))

print()
print("RESOLVED:")
for original, resolved, chosen_name, chosen_num in resolved_details:
    print(f"{original}  ->  {resolved}  ({chosen_name} #{chosen_num})")

if unresolved:
    print()
    print("UNRESOLVED:")
    for item in unresolved:
        print(item)
