from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
COMPILED_DEFAULT = ROOT / "data" / "compiled_cards" / "auto" / "compiled_cards_all.turn1_semantics.json"
CARDS_CSV_DEFAULT = ROOT / "data" / "all_cards.csv"
GOAL_FINDER_SCRIPT = ROOT / "scripts" / "run_turn1_goal_finder_strict.py"
COVERAGE_SCRIPT = ROOT / "scripts" / "analyze_turn1_runtime_coverage.py"
WORK_DIR = ROOT / "data" / "reports" / "turn1_access_frontend"
TEMP_DECK_DIR = ROOT / "data" / "decks" / "_frontend_temp"

TRUSTED_STATUSES = {
    "trusted_runtime",
    "trusted_with_runtime_guards",
    "trusted_global_rule",
    "trusted_from_compiled_ops",
}

EXAMPLE_MEGA_LUCARIO = """Pokémon: 8
1 Meowth ex POR 62
2 Solrock MEG 75
2 Makuhita MEG 72
1 Mega Zygarde ex POR 47
3 Mega Lucario ex MEG 77
2 Lunatone MEG 74
3 Riolu MEG 76
2 Hariyama MEG 73

Trainer: 14
3 Poké Pad ASC 198
4 Lillie's Determination MEG 119
2 Air Balloon BLK 79
2 Gravity Mountain SSP 177
1 Boss's Orders MEG 114
4 Fighting Gong MEG 116
3 Night Stretcher SFA 61
1 Switch MEG 130
4 Ultra Ball MEG 131
2 Wally's Compassion MEG 132
2 Judge POR 76
1 Unfair Stamp TWM 165
1 Core Memory POR 70
4 Premium Power Pro MEG 124

Energy: 1
10 Basic {F} Energy MEE 6

Total Cards: 60
"""


# ------------------------------------------------------------
# Small parsing helpers
# ------------------------------------------------------------
def clean_card_name(name: str) -> str:
    type_map = {
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
    out = str(name or "").replace("\ufeff", "").strip()
    for symbol, word in type_map.items():
        out = out.replace(symbol, word)
    return " ".join(out.split())


def parse_deck_names(decklist_text: str) -> pd.DataFrame:
    """Parse enough of a PTCGL-style decklist to populate the goal selector.

    This intentionally does not try to resolve official IDs. The strict simulator
    does the real deck resolution. This just extracts user-facing card names.
    """
    rows: list[dict[str, Any]] = []
    current_section = "Unknown"

    for raw_line in str(decklist_text or "").splitlines():
        line = raw_line.strip().replace("\ufeff", "")
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("pokémon:") or lower.startswith("pokemon:"):
            current_section = "Pokémon"
            continue
        if lower.startswith("trainer:"):
            current_section = "Trainer"
            continue
        if lower.startswith("energy:"):
            current_section = "Energy"
            continue
        if lower.startswith("total cards"):
            continue

        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue

        qty = int(parts[0])

        # PTCGL export style: qty + name + set_code + collector_number.
        # Plain simulator style: qty + api_id. For plain IDs, use the ID as label.
        if len(parts) >= 4 and parts[-2].isalnum():
            name = clean_card_name(" ".join(parts[1:-2]))
            set_code = parts[-2]
            number = parts[-1]
        else:
            name = clean_card_name(" ".join(parts[1:]))
            set_code = ""
            number = ""

        rows.append(
            {
                "count": qty,
                "card_name": name,
                "section": current_section,
                "set_code": set_code,
                "number": number,
                "raw_line": line,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["count", "card_name", "section", "set_code", "number", "raw_line"])

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["card_name", "section"], as_index=False)
        .agg(count=("count", "sum"), examples=("raw_line", lambda s: "; ".join(list(s)[:3])))
        .sort_values(["section", "card_name"])
    )
    return grouped


def pct(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}%"


def run_cmd(args: list[str], timeout: int | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def str_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


# ------------------------------------------------------------
# Coverage check
# ------------------------------------------------------------
def run_coverage_check(deck_path: Path, prefix: str) -> tuple[dict[str, Any], pd.DataFrame, str]:
    if not COVERAGE_SCRIPT.exists():
        return {}, pd.DataFrame(), f"Coverage script not found: {COVERAGE_SCRIPT}"

    args = [
        sys.executable,
        str(COVERAGE_SCRIPT),
        "--compiled",
        str(COMPILED_DEFAULT),
        "--cards-csv",
        str(CARDS_CSV_DEFAULT),
        "--decklist",
        str(deck_path),
        "--outdir",
        str(WORK_DIR / "coverage"),
        "--prefix",
        prefix,
    ]
    code, stdout, stderr = run_cmd(args, timeout=180)
    if code != 0:
        return {}, pd.DataFrame(), (stderr or stdout or "Coverage command failed.")

    try:
        summary = json.loads(stdout[stdout.find("{") : stdout.rfind("}") + 1])
    except Exception:
        summary = {}

    audit_path = WORK_DIR / "coverage" / f"{prefix}_current_deck_runtime_audit.csv"
    audit_df = read_csv_if_exists(audit_path)
    if audit_df.empty:
        return summary, pd.DataFrame(), ""

    unsupported = audit_df[
        (audit_df.get("turn1_relevant", "").map(str_true))
        & (~audit_df.get("trust_status", "").isin(TRUSTED_STATUSES))
    ].copy()

    keep_cols = [
        c
        for c in ["card_name", "effect_name", "trust_status", "families", "ops", "text"]
        if c in unsupported.columns
    ]
    return summary, unsupported[keep_cols], ""


# ------------------------------------------------------------
# Batch simulation aggregation
# ------------------------------------------------------------
@dataclass
class ScenarioAgg:
    trials: int = 0
    weighted_final_percent_sum: float = 0.0
    weighted_raw_percent_sum: float = 0.0
    natural_percent: float | None = None
    missing_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.missing_counts is None:
            self.missing_counts = Counter()

    @property
    def final_percent(self) -> float | None:
        if self.trials <= 0:
            return None
        return self.weighted_final_percent_sum / self.trials

    @property
    def raw_percent(self) -> float | None:
        if self.trials <= 0:
            return None
        return self.weighted_raw_percent_sum / self.trials

    @property
    def ci95(self) -> tuple[float, float] | None:
        if self.trials <= 0 or self.final_percent is None:
            return None
        p = max(0.0, min(1.0, self.final_percent / 100.0))
        se = math.sqrt(max(0.0, p * (1.0 - p) / self.trials))
        low = max(0.0, (p - 1.96 * se) * 100.0)
        high = min(100.0, (p + 1.96 * se) * 100.0)
        return low, high


def update_line_counts(lines_df: pd.DataFrame, line_counts: dict[str, Counter[str]], batch_trials: int) -> None:
    if lines_df.empty:
        return

    # Expected columns from current CLI: going, line, count, percent.
    # Be defensive because the CLI report may evolve.
    going_col = "going" if "going" in lines_df.columns else None
    line_col = "line" if "line" in lines_df.columns else None
    if line_col is None:
        for candidate in ["sequence", "action_line", "actions", "label"]:
            if candidate in lines_df.columns:
                line_col = candidate
                break
    if line_col is None:
        return

    for _, row in lines_df.iterrows():
        going = str(row.get(going_col, "both") if going_col else "both")
        line = str(row.get(line_col, "")).strip()
        if not line:
            continue
        count_val = row.get("count", None)
        try:
            count = int(round(float(count_val)))
        except Exception:
            try:
                count = int(round(float(row.get("percent", 0.0)) / 100.0 * batch_trials))
            except Exception:
                count = 1
        line_counts[going][line] += max(1, count)


def line_counts_to_df(line_counts: Counter[str], total_trials: int, limit: int = 15) -> pd.DataFrame:
    rows = []
    denom = max(1, total_trials)
    for line, count in line_counts.most_common(limit):
        rows.append({"line": line, "count": count, "percent_of_trials": count / denom * 100.0})
    return pd.DataFrame(rows)


def missing_counts_to_df(counter: Counter[str], total_trials: int) -> pd.DataFrame:
    denom = max(1, total_trials)
    rows = [
        {"card": card, "missing_count": count, "percent_of_trials": count / denom * 100.0}
        for card, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def run_simulation_batch(
    deck_path: Path,
    goal: str,
    goal_mode: str,
    goal_zone: str,
    going: str,
    batch_trials: int,
    seed: int,
    max_actions: int,
    batch_prefix: str,
    example_lines: int,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    out_json = WORK_DIR / f"{batch_prefix}.json"
    out_summary = WORK_DIR / f"{batch_prefix}_summary.csv"
    out_lines = WORK_DIR / f"{batch_prefix}_lines.csv"

    args = [
        sys.executable,
        str(GOAL_FINDER_SCRIPT),
        "--compiled",
        str(COMPILED_DEFAULT),
        "--decklist",
        str(deck_path),
        "--goal",
        goal,
        "--goal-mode",
        goal_mode,
        "--goal-zone",
        goal_zone,
        "--going",
        going,
        "--trials",
        str(batch_trials),
        "--seed",
        str(seed),
        "--max-actions",
        str(max_actions),
        "--chain-search",
        "--example-lines",
        str(example_lines),
        "--out",
        str(out_json),
        "--csv-out",
        str(out_summary),
        "--lines-csv",
        str(out_lines),
    ]

    code, stdout, stderr = run_cmd(args, timeout=300)
    if code != 0:
        return {}, pd.DataFrame(), stderr or stdout or "Simulation command failed."

    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        try:
            data = json.loads(stdout[stdout.find("{") : stdout.rfind("}") + 1])
        except Exception:
            data = {}

    lines_df = read_csv_if_exists(out_lines)
    return data, lines_df, ""


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Turn 1 Access Simulator", page_icon="🎲", layout="wide")

st.title("Turn 1 Access Simulator")
st.caption(
    "Estimate the probability of accessing selected cards by Turn 1. "
    "This is card-access probability, not full attack-ready / evolution-ready board simulation."
)

missing_files = [
    str(p.relative_to(ROOT))
    for p in [COMPILED_DEFAULT, GOAL_FINDER_SCRIPT]
    if not p.exists()
]
if missing_files:
    st.error("Required files are missing: " + ", ".join(missing_files))
    st.stop()

with st.sidebar:
    st.header("Simulation settings")
    total_trials = st.number_input("Trials", min_value=100, max_value=100000, value=1000, step=100)
    batch_size = st.number_input("Batch size", min_value=25, max_value=5000, value=100, step=25)
    seed = st.number_input("Seed", min_value=1, max_value=9999999, value=1, step=1)
    max_actions = st.slider("Max actions", min_value=1, max_value=20, value=8)
    example_lines = st.slider("Success lines per batch", min_value=10, max_value=500, value=100, step=10)
    run_coverage = st.checkbox("Run coverage check first", value=True)
    allow_unsupported = st.checkbox("Allow run with unsupported coverage rows", value=False)

st.subheader("1. Paste decklist")
decklist_text = st.text_area(
    "Decklist",
    value=EXAMPLE_MEGA_LUCARIO,
    height=360,
    help="PTCGL-style exports are supported. Plain 'count card_id' lines also work if your simulator can resolve them.",
)

parsed_cards = parse_deck_names(decklist_text)
parsed_count = int(parsed_cards["count"].sum()) if not parsed_cards.empty else 0

col_count, col_unique = st.columns(2)
col_count.metric("Parsed cards", parsed_count)
col_unique.metric("Unique names", len(parsed_cards))

if parsed_count != 60:
    st.warning(f"Parsed deck count is {parsed_count}, not 60. The simulator may still fail until the decklist is corrected.")

with st.expander("Parsed card names", expanded=False):
    st.dataframe(parsed_cards, use_container_width=True, hide_index=True)

st.subheader("2. Choose goal")
name_options = parsed_cards["card_name"].tolist() if not parsed_cards.empty else []
default_goal = [name for name in ["Lunatone", "Solrock"] if name in name_options]

g1, g2, g3 = st.columns([1.3, 0.7, 0.8])
with g1:
    selected_goal_cards = st.multiselect("Cards to access", options=name_options, default=default_goal)
with g2:
    goal_mode_label = st.radio("Goal mode", options=["All selected", "Any selected"], horizontal=False)
with g3:
    goal_zone = st.selectbox("Goal zone", options=["hand_or_in_play", "hand", "in_play", "accessed"], index=0)

going = st.radio("Going", options=["both", "first", "second"], horizontal=True)
goal_mode = "all" if goal_mode_label == "All selected" else "any"
goal = ", ".join(selected_goal_cards)

run_button = st.button("Run Turn 1 access simulation", type="primary", use_container_width=True)

if run_button:
    if not selected_goal_cards:
        st.error("Select at least one goal card.")
        st.stop()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DECK_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    safe_goal = "_".join("".join(ch.lower() if ch.isalnum() else "_" for ch in c).strip("_") for c in selected_goal_cards)
    prefix = f"turn1_access_{timestamp}_{safe_goal[:60]}"
    deck_path = TEMP_DECK_DIR / f"{prefix}.txt"
    deck_path.write_text(decklist_text, encoding="utf-8")

    if run_coverage:
        with st.spinner("Checking runtime coverage for this deck..."):
            coverage_summary, unsupported_df, coverage_error = run_coverage_check(deck_path, prefix)

        if coverage_error:
            st.warning(f"Coverage check could not complete: {coverage_error}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Deck effects scanned", coverage_summary.get("deck_effects", "—"))
            c2.metric("Unsupported relevant effects", len(unsupported_df))
            c3.metric("Coverage version", coverage_summary.get("version", "—"))

            if not unsupported_df.empty:
                st.warning("This deck has unsupported Turn-1-relevant effects. Review before trusting the result.")
                st.dataframe(unsupported_df, use_container_width=True, hide_index=True)
                if not allow_unsupported:
                    st.stop()
            else:
                st.success("Coverage check is clean for Turn-1 card-access effects.")

    st.subheader("3. Live result")
    progress = st.progress(0)
    status = st.empty()
    metric_area = st.container()
    chart_area = st.empty()
    missing_area = st.container()
    lines_area = st.container()

    scenario_aggs: dict[str, ScenarioAgg] = defaultdict(ScenarioAgg)
    line_counts: dict[str, Counter[str]] = defaultdict(Counter)
    history_rows: list[dict[str, Any]] = []

    total_trials = int(total_trials)
    batch_size = int(batch_size)
    batches = math.ceil(total_trials / batch_size)
    completed = 0

    for batch_idx in range(batches):
        this_batch = min(batch_size, total_trials - completed)
        batch_seed = int(seed) + batch_idx * 10007
        batch_prefix = f"{prefix}_batch_{batch_idx + 1:04d}"

        status.info(f"Running batch {batch_idx + 1}/{batches} ({completed + this_batch}/{total_trials} trials)...")
        data, lines_df, sim_error = run_simulation_batch(
            deck_path=deck_path,
            goal=goal,
            goal_mode=goal_mode,
            goal_zone=goal_zone,
            going=going,
            batch_trials=this_batch,
            seed=batch_seed,
            max_actions=int(max_actions),
            batch_prefix=batch_prefix,
            example_lines=int(example_lines),
        )

        if sim_error:
            st.error(sim_error)
            st.stop()

        if not data.get("passed") or not data.get("scenarios"):
            st.error("The simulator did not return scenarios. Check that the decklist and selected goal names resolve to cards.")
            st.json(data)
            st.stop()

        completed += this_batch
        update_line_counts(lines_df, line_counts, this_batch)

        for scenario in data.get("scenarios", []):
            going_key = str(scenario.get("going", "unknown"))
            agg = scenario_aggs[going_key]
            agg.trials += this_batch
            agg.weighted_final_percent_sum += float(scenario.get("final_exact_plus_sim_percent", 0.0)) * this_batch
            agg.weighted_raw_percent_sum += float(scenario.get("raw_sim_percent", 0.0)) * this_batch
            agg.natural_percent = float(scenario.get("exact_seen_by_draw_for_turn_percent", 0.0))
            for miss in scenario.get("top_missing", []) or []:
                req = str(miss.get("requirement", "Unknown"))
                try:
                    count = int(miss.get("count", 0))
                except Exception:
                    count = 0
                agg.missing_counts[req] += count

        for going_key, agg in scenario_aggs.items():
            history_rows.append(
                {
                    "completed_trials": completed,
                    "going": going_key,
                    "success_percent": agg.final_percent,
                    "raw_sim_percent": agg.raw_percent,
                }
            )

        progress.progress(completed / total_trials)

        with metric_area:
            cols = st.columns(max(1, len(scenario_aggs)))
            for col, (going_key, agg) in zip(cols, sorted(scenario_aggs.items())):
                ci = agg.ci95
                ci_text = f"95% CI {ci[0]:.1f}%–{ci[1]:.1f}%" if ci else ""
                col.metric(
                    label=f"Going {going_key}",
                    value=pct(agg.final_percent, 1),
                    delta=ci_text,
                )
                col.caption(f"Natural seen by draw: {pct(agg.natural_percent, 1)}")

        if history_rows:
            hist_df = pd.DataFrame(history_rows)
            pivot = hist_df.pivot_table(index="completed_trials", columns="going", values="success_percent", aggfunc="last")
            chart_area.line_chart(pivot)

        with missing_area:
            st.markdown("#### Top missing cards on failed trials")
            miss_cols = st.columns(max(1, len(scenario_aggs)))
            for col, (going_key, agg) in zip(miss_cols, sorted(scenario_aggs.items())):
                col.markdown(f"**Going {going_key}**")
                miss_df = missing_counts_to_df(agg.missing_counts, agg.trials)
                if miss_df.empty:
                    col.caption("No missing-card data yet.")
                else:
                    col.dataframe(miss_df, use_container_width=True, hide_index=True)

        with lines_area:
            st.markdown("#### Success lines seen so far")
            line_cols = st.columns(max(1, len(scenario_aggs)))
            for col, going_key in zip(line_cols, sorted(scenario_aggs.keys())):
                col.markdown(f"**Going {going_key}**")
                df = line_counts_to_df(line_counts.get(going_key, Counter()), scenario_aggs[going_key].trials, limit=20)
                if df.empty:
                    col.caption("No line details yet.")
                else:
                    col.dataframe(df, use_container_width=True, hide_index=True)

    status.success(f"Done: {completed:,} trials completed.")
    st.caption("Result is for Turn-1 card access only. It does not yet validate full evolution, attack, retreat, or board-state sequencing.")
