# TURN1_RUNTIME_CONTROLS_V47
from __future__ import annotations

import json
import math
import subprocess

# ---------------------------------------------------------------------
# TURN1_FRONTEND_SUBPROCESS_TIMEOUT_GUARD_V45
# ---------------------------------------------------------------------
# Prevent Streamlit from hanging forever while Automatic Goal Finder runs.
#
# Defaults can be overridden in PowerShell before launching Streamlit:
#   $env:TURN1_STREAMLIT_GOAL_TIMEOUT_SECONDS = "600"
#   $env:TURN1_STREAMLIT_MAX_TRIALS = "200"
#   $env:TURN1_STREAMLIT_MAX_ACTIONS = "8"

_ORIG_SUBPROCESS_RUN_TURN1_V45 = subprocess.run


def _turn1_v45_int_env(name, default):
    import os as _os

    try:
        return int(_os.environ.get(name, str(default)))
    except Exception:
        return default


def _turn1_v45_is_goal_finder_cmd(cmd):
    if isinstance(cmd, (list, tuple)):
        joined = " ".join(str(x) for x in cmd).lower()
    else:
        joined = str(cmd).lower()

    return "run_turn1_goal_finder" in joined


def _turn1_v45_cap_flag(cmd, flag, cap):
    if not isinstance(cmd, list):
        return cmd, None

    if flag not in cmd:
        return cmd, None

    idx = cmd.index(flag)

    if idx + 1 >= len(cmd):
        return cmd, None

    old = cmd[idx + 1]

    try:
        old_int = int(old)
    except Exception:
        return cmd, None

    if old_int > cap:
        cmd[idx + 1] = str(cap)
        return cmd, (flag, old_int, cap)

    return cmd, None


def _turn1_v45_capped_goal_finder_cmd(cmd):
    if not isinstance(cmd, (list, tuple)):
        return cmd, []

    cmd = list(cmd)

    if not _turn1_v45_is_goal_finder_cmd(cmd):
        return cmd, []

    changes = []

    max_trials = _turn1_v45_int_env("TURN1_STREAMLIT_MAX_TRIALS", 100)
    max_actions = _turn1_v45_int_env("TURN1_STREAMLIT_MAX_ACTIONS", 6)
    max_workers = _turn1_v45_int_env("TURN1_STREAMLIT_MAX_WORKERS", 2)

    "--trials", str(int(trials)),
    if change:
        changes.append(change)

    cmd, change = _turn1_v45_cap_flag(cmd, "--max-actions", max_actions)
    if change:
        changes.append(change)
    "--workers", str(int(workers)),
    if change:
        changes.append(change)

    return cmd, changes


def _turn1_v45_timeout_completed_process(cmd, timeout_seconds, exc, cap_changes):
    stdout = getattr(exc, "stdout", "") or ""
    stderr = getattr(exc, "stderr", "") or ""

    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")

    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")

    cap_msg = ""

    if cap_changes:
        cap_lines = [
            f"{flag}: requested {old}, capped to {new}"
            for flag, old, new in cap_changes
        ]
        cap_msg = "\nFrontend safety caps applied:\n" + "\n".join(cap_lines)

    stderr = (
        str(stderr)
        + "\n\nTurn 1 Automatic Goal Finder timed out in the Streamlit frontend."
        + f"\nTimeout: {timeout_seconds} seconds."
        + cap_msg
        + "\n\nTry fewer trials, max actions 6, and chain-search off/on depending on the test."
    )

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=124,
        stdout=stdout,
        stderr=stderr,
    )


def _turn1_v45_subprocess_run_with_timeout(*args, **kwargs):
    timeout_seconds = _turn1_v45_int_env("TURN1_STREAMLIT_GOAL_TIMEOUT_SECONDS", 600)

    cap_changes = []

    if args:
        cmd, cap_changes = _turn1_v45_capped_goal_finder_cmd(args[0])
        args = (cmd,) + args[1:]
    elif "args" in kwargs:
        cmd, cap_changes = _turn1_v45_capped_goal_finder_cmd(kwargs["args"])
        kwargs["args"] = cmd
    else:
        cmd = None

    if _turn1_v45_is_goal_finder_cmd(cmd):
        kwargs.setdefault("timeout", timeout_seconds)

    try:
        return _ORIG_SUBPROCESS_RUN_TURN1_V45(*args, **kwargs)
    except subprocess.TimeoutExpired as exc:
        return _turn1_v45_timeout_completed_process(
            cmd=cmd,
            timeout_seconds=timeout_seconds,
            exc=exc,
            cap_changes=cap_changes,
        )


subprocess.run = _turn1_v45_subprocess_run_with_timeout


import os
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from src.probability import p_dnf_statement_given_legal_opening
except Exception:
    p_dnf_statement_given_legal_opening = None


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _pct_fraction(p: float | None, decimals: int = 2) -> str:
    if p is None:
        return "—"
    try:
        return f"{100 * float(p):.{decimals}f}%"
    except Exception:
        return "—"


def _pct_percent(p: float | None, decimals: int = 2) -> str:
    if p is None:
        return "—"
    try:
        return f"{float(p):.{decimals}f}%"
    except Exception:
        return "—"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def _card_attr(card: Any, name: str, default: Any = None) -> Any:
    if isinstance(card, dict):
        return card.get(name, default)
    return getattr(card, name, default)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reports_dir() -> Path:
    path = _repo_root() / "data" / "reports" / "streamlit_turn1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compiled_semantics_path() -> Path:
    # TURN1_GZIP_COMPILED_SEMANTICS_PATH
    plain = _repo_root() / "data" / "compiled_cards" / "auto" / "compiled_cards_all.turn1_semantics.json"
    if plain.exists():
        return plain

    gz = Path(str(plain) + ".gz")
    if gz.exists():
        return gz

    return plain



def _turn1_cmd_arg_value(cmd: list[Any], flag: str, default: str = "") -> str:
    try:
        idx = list(cmd).index(flag)
        if idx + 1 < len(cmd):
            return str(cmd[idx + 1])
    except Exception:
        pass
    return default


def _run_goal_finder_subprocess_with_progress(cmd: list[Any], *, cwd: str) -> subprocess.CompletedProcess:
    timeout_seconds = _turn1_v45_int_env("TURN1_STREAMLIT_GOAL_TIMEOUT_SECONDS", 600)

    trials_label = _turn1_cmd_arg_value(cmd, "--trials", "?")
    going_label = _turn1_cmd_arg_value(cmd, "--going", "?")
    workers_label = _turn1_cmd_arg_value(cmd, "--workers", "?")
    actions_label = _turn1_cmd_arg_value(cmd, "--max-actions", "?")
    progress_json_raw = _turn1_cmd_arg_value(cmd, "--progress-json", "")

    progress_json_path = Path(progress_json_raw) if progress_json_raw else None
    if progress_json_path is not None:
        try:
            progress_json_path.unlink(missing_ok=True)
        except Exception:
            pass

    progress = st.progress(
        0,
        text=(
            "Starting automatic goal finder... "
            f"trials={trials_label}, going={going_label}, "
            f"workers={workers_label}, max actions={actions_label}"
        ),
    )
    status = st.empty()

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )

    stdout = ""
    stderr = ""

    last_completed = 0
    last_total = 0

    def read_progress_payload() -> dict[str, Any] | None:
        if progress_json_path is None or not progress_json_path.exists():
            return None
        try:
            return json.loads(progress_json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    try:
        while proc.poll() is None:
            elapsed = time.time() - start

            if timeout_seconds and elapsed >= timeout_seconds:
                proc.kill()
                stdout, stderr = proc.communicate()
                stderr = (
                    str(stderr or "")
                    + "\n\nTurn 1 Automatic Goal Finder timed out in the Streamlit frontend."
                    + f"\nTimeout: {timeout_seconds} seconds."
                    + "\n\nTry fewer trials, max actions 6, and chain-search off/on depending on the test."
                )
                progress.progress(100, text=f"Goal finder timed out after {elapsed:.1f}s.")
                status.error("Automatic goal finder timed out.")
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=124,
                    stdout=stdout,
                    stderr=stderr,
                )

            payload = read_progress_payload()
            if payload:
                try:
                    completed = int(payload.get("completed_trials", 0) or 0)
                    total = int(payload.get("total_trials", 0) or 0)
                except Exception:
                    completed, total = 0, 0

                if total > 0:
                    last_completed = max(last_completed, completed)
                    last_total = max(last_total, total)
                    pct = min(99, max(0, int((last_completed / last_total) * 100)))
                    progress.progress(
                        pct,
                        text=(
                            f"Running trials: {last_completed}/{last_total} complete "
                            f"({pct}%). Elapsed {elapsed:.1f}s."
                        ),
                    )
                    status.caption(
                        f"Going={payload.get('going') or going_label}; "
                        f"workers={workers_label}; max actions={actions_label}"
                    )
                else:
                    progress.progress(0, text=f"Preparing trial progress... elapsed {elapsed:.1f}s.")
                    status.caption("Waiting for backend trial counter.")
            else:
                progress.progress(0, text=f"Preparing simulation... elapsed {elapsed:.1f}s.")
                status.caption("Waiting for backend trial counter.")

            time.sleep(0.25)

        stdout, stderr = proc.communicate()
        elapsed = time.time() - start

        payload = read_progress_payload()
        if payload:
            try:
                completed = int(payload.get("completed_trials", last_completed) or last_completed)
                total = int(payload.get("total_trials", last_total) or last_total)
            except Exception:
                completed, total = last_completed, last_total

            if total > 0:
                progress.progress(
                    100 if proc.returncode == 0 else min(99, int((completed / total) * 100)),
                    text=f"Trials complete: {completed}/{total}. Elapsed {elapsed:.1f}s.",
                )
            else:
                progress.progress(100, text=f"Goal finder complete in {elapsed:.1f}s.")
        else:
            progress.progress(100, text=f"Goal finder complete in {elapsed:.1f}s.")

        if proc.returncode == 0:
            status.success("Automatic goal finder finished.")
        else:
            status.warning("Automatic goal finder returned a non-zero exit code.")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    except Exception as exc:
        try:
            proc.kill()
            stdout, stderr = proc.communicate()
        except Exception:
            pass

        elapsed = time.time() - start
        progress.progress(100, text=f"Goal finder failed after {elapsed:.1f}s.")
        status.error("Automatic goal finder failed.")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout=stdout,
            stderr=(str(stderr or "") + f"\n\nFrontend progress runner error: {exc}"),
        )

def _strict_goal_finder_path() -> Path:
    return _repo_root() / "scripts" / "run_turn1_goal_finder_strict.py"


# ---------------------------------------------------------------------
# Deck parsing helpers
# ---------------------------------------------------------------------

def _deck_options(deck: Any, card_odds_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    if deck is not None:
        try:
            for i, card in enumerate(deck):
                label = (
                    _card_attr(card, "label")
                    or _card_attr(card, "card")
                    or _card_attr(card, "name")
                    or f"Card {i + 1}"
                )
                name = _card_attr(card, "name", label)
                count = _safe_int(_card_attr(card, "count", 0))
                supertype = _card_attr(card, "supertype", "Unknown") or "Unknown"
                is_basic = bool(_card_attr(card, "is_basic_pokemon", False))
                image_url = (
                    _card_attr(card, "image_large_url")
                    or _card_attr(card, "image_url")
                    or _card_attr(card, "image_small")
                    or None
                )

                if count <= 0:
                    continue

                options.append(
                    {
                        "key": f"{i}:{label}",
                        "label": str(label),
                        "name": str(name),
                        "count": count,
                        "supertype": str(supertype),
                        "subtypes": str(_card_attr(card, "subtypes", "") or ""),
                        "is_basic": is_basic,
                        "image_url": image_url,
                    }
                )
        except Exception:
            options = []

    if options:
        return options

    if card_odds_df is not None and not card_odds_df.empty:
        df = card_odds_df.copy()
        for i, row in df.reset_index(drop=True).iterrows():
            label = row.get("card", row.get("name", f"Card {i + 1}"))
            count = _safe_int(row.get("count", 0))
            if count <= 0:
                continue

            options.append(
                {
                    "key": f"{i}:{label}",
                    "label": str(label),
                    "name": str(row.get("name", label)),
                    "count": count,
                    "supertype": str(row.get("supertype", "Unknown") or "Unknown"),
                    "subtypes": str(row.get("subtypes", "") or row.get("subtype", "") or ""),
                    "is_basic": bool(row.get("is_basic_pokemon", False)),
                    "image_url": row.get("image_large_url") or row.get("image_url") or row.get("image_small"),
                }
            )

    return options


def _default_goal_labels(options: list[dict[str, Any]]) -> list[str]:
    labels = [opt["label"] for opt in options]

    def find(part: str) -> str | None:
        needle = part.lower()
        for opt in options:
            text = f"{opt['label']} {opt['name']}".lower()
            if needle in text:
                return opt["label"]
        return None

    lunatone = find("lunatone")
    solrock = find("solrock")
    if lunatone and solrock:
        return [lunatone, solrock]

    non_energy = [opt["label"] for opt in options if opt.get("supertype", "").lower() != "energy"]
    return non_energy[:2] if len(non_energy) >= 2 else labels[: min(2, len(labels))]


# ---------------------------------------------------------------------
# Baseline exact math
# ---------------------------------------------------------------------

def _fallback_unconditioned_probability(deck_size: int, draw_n: int, counts: list[int], mode: str) -> float:
    draw_n = max(0, min(draw_n, deck_size))
    total = _comb(deck_size, draw_n)
    if total <= 0 or not counts:
        return 0.0

    counts = [max(0, int(c)) for c in counts]

    if mode == "any":
        success_total = sum(counts)
        return 1.0 - (_comb(deck_size - success_total, draw_n) / total)

    probability = 0.0
    n = len(counts)

    for r in range(0, n + 1):
        sign = -1 if r % 2 else 1
        for subset in combinations(range(n), r):
            removed = sum(counts[i] for i in subset)
            probability += sign * (_comb(deck_size - removed, draw_n) / total)

    return max(0.0, min(1.0, probability))


def _exact_legal_opening_probability(
    *,
    deck_size: int,
    basic_count: int,
    selected_cards: list[dict[str, Any]],
    mode: str,
    include_turn_draw: bool,
) -> float:
    counts = [int(card["count"]) for card in selected_cards]
    basics = [bool(card["is_basic"]) for card in selected_cards]

    if not counts:
        return 0.0

    if p_dnf_statement_given_legal_opening is not None:
        try:
            if mode == "all":
                success_routes = [list(range(len(counts)))]
            else:
                success_routes = [[i] for i in range(len(counts))]

            return float(
                p_dnf_statement_given_legal_opening(
                    deck_size=deck_size,
                    basic_count=basic_count,
                    card_counts=counts,
                    card_is_basic=basics,
                    success_routes=success_routes,
                    include_turn_draw=include_turn_draw,
                )
            )
        except Exception:
            pass

    draw_n = 8 if include_turn_draw else 7
    return _fallback_unconditioned_probability(deck_size, draw_n, counts, mode)




# ---------------------------------------------------------------------
# TURN1_EXACT_REQUIRED_COPIES_V17
# ---------------------------------------------------------------------

def _goal_satisfied_by_counts(draw_counts: list[int], min_counts: list[int], mode: str) -> bool:
    if not draw_counts or not min_counts:
        return False

    if mode == "any":
        return any(x >= req for x, req in zip(draw_counts, min_counts))

    return all(x >= req for x, req in zip(draw_counts, min_counts))


def _iter_selected_count_vectors(card_counts: list[int], hand_size: int):
    """
    Yield possible draw counts for selected goal cards.

    Example:
      selected card counts = [2, 4]
      hand size = 7

    yields vectors like:
      [0, 0], [0, 1], ..., [2, 4]
    but never vectors whose sum exceeds hand_size.
    """
    out: list[int] = []

    def rec(i: int, remaining: int):
        if i == len(card_counts):
            yield list(out)
            return

        max_take = min(card_counts[i], remaining)

        for x in range(max_take + 1):
            out.append(x)
            yield from rec(i + 1, remaining - x)
            out.pop()

    yield from rec(0, hand_size)


def _exact_legal_opening_probability_required_copies(
    *,
    deck_size: int,
    basic_count: int,
    selected_cards: list[dict[str, Any]],
    mode: str,
    include_turn_draw: bool,
) -> float:
    """
    Exact natural-access probability with required-copy support.

    This answers:

      P(goal is naturally seen | opening 7 is legal)

    For include_turn_draw=True, it handles:
      opening 7 conditioned on having a Basic Pokémon,
      then one natural draw from the remaining deck.

    This supports goals like:
      - 2x Basic Fighting Energy
      - 2x Raging Bolt ex
      - 1x Raging Bolt ex + 1x Crispin
    """
    if not selected_cards:
        return 0.0

    deck_size = int(deck_size)
    basic_count = int(basic_count)

    card_counts = [int(card.get("count", 0)) for card in selected_cards]
    min_counts = [
        max(1, int(card.get("required_count", card.get("min_count", 1))))
        for card in selected_cards
    ]
    card_is_basic = [bool(card.get("is_basic", card.get("is_basic_pokemon", False))) for card in selected_cards]

    if any(req > cnt for req, cnt in zip(min_counts, card_counts)):
        return 0.0

    selected_total = sum(card_counts)
    selected_basic_total = sum(cnt for cnt, is_basic in zip(card_counts, card_is_basic) if is_basic)

    other_basic = max(0, basic_count - selected_basic_total)
    other_nonbasic = max(0, deck_size - selected_total - other_basic)

    opening_size = 7

    legal_opening_count = _comb(deck_size, opening_size) - _comb(deck_size - basic_count, opening_size)

    if legal_opening_count <= 0:
        return 0.0

    # ------------------------------------------------------------------
    # Opening hand only
    # ------------------------------------------------------------------
    if not include_turn_draw:
        numerator = 0

        for selected_draws in _iter_selected_count_vectors(card_counts, opening_size):
            selected_sum = sum(selected_draws)
            remaining_slots = opening_size - selected_sum

            for other_basic_draws in range(min(other_basic, remaining_slots) + 1):
                other_nonbasic_draws = remaining_slots - other_basic_draws

                if other_nonbasic_draws < 0 or other_nonbasic_draws > other_nonbasic:
                    continue

                basic_seen = other_basic_draws + sum(
                    x for x, is_basic in zip(selected_draws, card_is_basic) if is_basic
                )

                if basic_seen <= 0:
                    continue

                if not _goal_satisfied_by_counts(selected_draws, min_counts, mode):
                    continue

                ways = 1

                for cnt, drawn in zip(card_counts, selected_draws):
                    ways *= _comb(cnt, drawn)

                ways *= _comb(other_basic, other_basic_draws)
                ways *= _comb(other_nonbasic, other_nonbasic_draws)

                numerator += ways

        return max(0.0, min(1.0, numerator / legal_opening_count))

    # ------------------------------------------------------------------
    # Opening hand + natural draw
    # ------------------------------------------------------------------
    numerator = 0
    denominator = legal_opening_count * max(0, deck_size - opening_size)

    if denominator <= 0:
        return 0.0

    for selected_opening in _iter_selected_count_vectors(card_counts, opening_size):
        selected_sum = sum(selected_opening)
        remaining_slots = opening_size - selected_sum

        for other_basic_opening in range(min(other_basic, remaining_slots) + 1):
            other_nonbasic_opening = remaining_slots - other_basic_opening

            if other_nonbasic_opening < 0 or other_nonbasic_opening > other_nonbasic:
                continue

            basic_seen = other_basic_opening + sum(
                x for x, is_basic in zip(selected_opening, card_is_basic) if is_basic
            )

            if basic_seen <= 0:
                continue

            opening_ways = 1

            for cnt, drawn in zip(card_counts, selected_opening):
                opening_ways *= _comb(cnt, drawn)

            opening_ways *= _comb(other_basic, other_basic_opening)
            opening_ways *= _comb(other_nonbasic, other_nonbasic_opening)

            # Draw one selected goal card.
            for i, cnt in enumerate(card_counts):
                remaining_card_copies = cnt - selected_opening[i]

                if remaining_card_copies <= 0:
                    continue

                final_counts = list(selected_opening)
                final_counts[i] += 1

                if _goal_satisfied_by_counts(final_counts, min_counts, mode):
                    numerator += opening_ways * remaining_card_copies

            # Draw one other Basic.
            remaining_other_basic = other_basic - other_basic_opening

            if remaining_other_basic > 0:
                final_counts = list(selected_opening)

                if _goal_satisfied_by_counts(final_counts, min_counts, mode):
                    numerator += opening_ways * remaining_other_basic

            # Draw one other non-Basic.
            remaining_other_nonbasic = other_nonbasic - other_nonbasic_opening

            if remaining_other_nonbasic > 0:
                final_counts = list(selected_opening)

                if _goal_satisfied_by_counts(final_counts, min_counts, mode):
                    numerator += opening_ways * remaining_other_nonbasic

    return max(0.0, min(1.0, numerator / denominator))


# Override the earlier exact function so the rest of the page automatically
# respects Required copies.
def _exact_legal_opening_probability(
    *,
    deck_size: int,
    basic_count: int,
    selected_cards: list[dict[str, Any]],
    mode: str,
    include_turn_draw: bool,
) -> float:
    return _exact_legal_opening_probability_required_copies(
        deck_size=deck_size,
        basic_count=basic_count,
        selected_cards=selected_cards,
        mode=mode,
        include_turn_draw=include_turn_draw,
    )


# ---------------------------------------------------------------------
# Goal images
# ---------------------------------------------------------------------

def _render_goal_images(selected_cards: list[dict[str, Any]]) -> None:
    st.markdown("### Goal cards")
    cols = st.columns(max(1, min(4, len(selected_cards))))

    for col, card in zip(cols, selected_cards):
        with col:
            if card.get("image_url"):
                st.image(card["image_url"], width=140)
            st.caption(f"{card['label']} ({card['count']}x)")


# ---------------------------------------------------------------------
# Goal-finder runtime helpers
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# TURN1_RUNTIME_LABEL_NORMALIZATION_V10
# ---------------------------------------------------------------------

def _runtime_label_for_decklist(label: str) -> str:
    """
    Convert UI labels like:
      Lunatone [MEG 74]
    into strict goal-finder decklist labels like:
      Lunatone MEG 74

    The frontend displays set info in brackets, but the backend resolver
    expects PTCGL-ish plain tokens.
    """
    import re

    raw = str(label or "").strip()

    m = re.match(r"^(.*?)\s*\[([A-Z0-9]+)\s+([A-Za-z0-9]+)\]\s*$", raw)
    if m:
        name, set_code, number = m.groups()
        return f"{name.strip()} {set_code.strip()} {number.strip()}"

    return raw


def _runtime_goal_name(label: str) -> str:
    """
    Convert UI labels like:
      Lunatone [MEG 74]
    into goal names like:
      Lunatone

    The goal finder does better when goals are canonical names rather than
    display labels with set brackets.
    """
    import re

    raw = str(label or "").strip()

    m = re.match(r"^(.*?)\s*\[[A-Z0-9]+\s+[A-Za-z0-9]+\]\s*$", raw)
    if m:
        return m.group(1).strip()

    # Also handle plain PTCGL style: "Lunatone MEG 74" -> "Lunatone"
    parts = raw.split()
    if len(parts) >= 3 and parts[-1].isdigit():
        return " ".join(parts[:-2]).strip()

    return raw


def _display_label_from_runtime_goal(goal_name: str, selected_labels: list[str]) -> str:
    """
    Map clean runtime goal names back to the UI label when possible.
    """
    low = str(goal_name or "").lower()
    for label in selected_labels:
        if low and low in str(label).lower():
            return label
    return str(goal_name)



# TURN1_EXACT_CARD_ID_DECKLIST_WRITER_V1
def _turn1_option_card_id_for_decklist_v1(opt: dict[str, Any]) -> str:
    if not isinstance(opt, dict):
        return ""

    for key in ("card_id", "id", "representative_card_id"):
        val = opt.get(key)
        if val:
            return str(val).strip()

    card = opt.get("card")
    if isinstance(card, dict):
        for key in ("representative_card_id", "id", "card_id"):
            val = card.get(key)
            if val:
                return str(val).strip()
        try:
            raw = ((card.get("sources") or {}).get("raw_card") or {})
            if raw.get("id"):
                return str(raw.get("id")).strip()
        except Exception:
            pass

    return ""


def _turn1_decklist_line_for_option_v1(opt: dict[str, Any]) -> str:
    count = int(opt["count"])
    label = _runtime_label_for_decklist(str(opt["label"]))
    card_id = _turn1_option_card_id_for_decklist_v1(opt)
    if card_id:
        return f"{count} {label} [{card_id}]"
    return f"{count} {label}"


def _write_decklist_file(options: list[dict[str, Any]], path: Path) -> None:
    lines = [_turn1_decklist_line_for_option_v1(opt) for opt in options]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_json_maybe(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None




# ---------------------------------------------------------------------
# TURN1_GOAL_MIN_COUNT_UI_V16
# ---------------------------------------------------------------------

def _goal_min_count_for_label(label: str, goal_quantities: dict[str, int] | None) -> int:
    try:
        return max(1, int((goal_quantities or {}).get(label, 1)))
    except Exception:
        return 1


def _goal_label_with_count(label: str, goal_quantities: dict[str, int] | None) -> str:
    q = _goal_min_count_for_label(label, goal_quantities)
    clean = _runtime_goal_name(label)
    return clean if q == 1 else f"{q}x {clean}"


def _goal_display_name_from_quantities(
    selected_labels: list[str],
    goal_quantities: dict[str, int] | None,
) -> str:
    return " + ".join(
        _goal_label_with_count(label, goal_quantities)
        for label in selected_labels
    )


def _write_goal_file(
    path: Path,
    selected_labels: list[str],
    goal_mode: str,
    goal_zone: str,
    goal_quantities: dict[str, int] | None,
) -> None:
    """
    Write a JSON goal file for the strict goal finder.

    This supports requirements like:
      2x Raging Bolt ex
      2x Basic Fighting Energy
      1x Raging Bolt ex + 1x Crispin

    The important backend field is min_count.
    """
    requirements = []

    for label in selected_labels:
        clean_name = _runtime_goal_name(label)
        min_count = _goal_min_count_for_label(label, goal_quantities)

        requirements.append(
            {
                "label": clean_name if min_count == 1 else f"{min_count}x {clean_name}",
                "options": [clean_name],
                "zone": goal_zone,
                "min_count": min_count,
            }
        )

    payload = {
        "name": _goal_display_name_from_quantities(selected_labels, goal_quantities),
        "mode": goal_mode,
        "requirements": requirements,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_goal_finder(
    *,
    all_options: list[dict[str, Any]],
    selected_labels: list[str],
    goal_mode: str,
    goal_zone: str,
    going: str,
    trials: int,
    seed: int,
    max_actions: int,
    example_lines: int,
    workers: int,
    complete_only: bool,
    chain_search: bool,
    goal_quantities: dict[str, int] | None = None,
    excluded_labels: list[str] | None = None,
) -> dict[str, Any]:
    reports_dir = _reports_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_goal = "_".join(label.split()[0] for label in selected_labels)[:80] or "goal"

    decklist_path = reports_dir / f"deck_{safe_goal}_{stamp}.txt"
    json_path = reports_dir / f"turn1_{safe_goal}_{stamp}.json"
    summary_csv_path = reports_dir / f"turn1_{safe_goal}_{stamp}_summary.csv"
    lines_csv_path = reports_dir / f"turn1_{safe_goal}_{stamp}_lines.csv"
    progress_json_path = reports_dir / f"goal_progress_{stamp}.json"
    goal_file_path = reports_dir / f"goal_{safe_goal}_{stamp}.json"

    _write_decklist_file(all_options, decklist_path)
    _write_goal_file(
        goal_file_path,
        selected_labels=selected_labels,
        goal_mode=goal_mode,
        goal_zone=goal_zone,
        goal_quantities=goal_quantities or {},
    )

    cmd = [
        sys.executable,
        str(_strict_goal_finder_path()),
        "--compiled",
        str(_compiled_semantics_path()),
        "--decklist",
        str(decklist_path),
        "--goal-file",
        str(goal_file_path),
        "--goal-name",
        _goal_display_name_from_quantities(selected_labels, goal_quantities or {}),
        "--goal-mode",
        goal_mode,
        "--goal-zone",
        goal_zone,
        "--going",
        going,
        "--trials", str(int(trials)),
        "--workers", str(int(workers)),
        "--seed",
        str(int(seed)),
        "--max-actions",
        str(int(max_actions)),
        "--example-lines",
        str(int(example_lines)),
        "--out",
        str(json_path),
        "--csv-out",
        str(summary_csv_path),
        "--lines-csv",
        str(lines_csv_path),
        "--progress-json",
        str(progress_json_path),
    ]

    if complete_only:
        cmd.append("--complete-only")
    if chain_search:
        cmd.append("--chain-search")

    excluded_labels = excluded_labels or []
    excluded_names = [
        _runtime_goal_name(label)
        for label in excluded_labels
        if str(label).strip()
    ]

    if excluded_names:
        cmd.extend(["--exclude-played", ", ".join(excluded_names)])

    start = time.time()
    proc = _run_goal_finder_subprocess_with_progress(
        cmd,
        cwd=str(_repo_root()),
    )
    elapsed = time.time() - start

    stdout_json = _parse_json_maybe(proc.stdout)
    report_json = None
    if json_path.exists():
        try:
            report_json = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            report_json = None

    return {
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_json": stdout_json,
        "report_json": report_json,
        "paths": {
            "decklist": str(decklist_path),
            "goal_file": str(goal_file_path),
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "lines_csv": str(lines_csv_path),
            "progress_json": str(progress_json_path),
        },
    }


def _scenario_summary_df(result: dict[str, Any]) -> pd.DataFrame:
    stdout_json = result.get("stdout_json") or {}
    scenarios = stdout_json.get("scenarios") or []

    rows = []
    for scenario in scenarios:
        ci = scenario.get("ci95_percent") or {}
        rows.append(
            {
                "Going": scenario.get("going"),
                "Final exact + sim": _pct_percent(scenario.get("final_exact_plus_sim_percent")),
                "Raw sim": _pct_percent(scenario.get("raw_sim_percent")),
                "Exact by natural draw": _pct_percent(scenario.get("exact_seen_by_draw_for_turn_percent")),
                "CI low": _pct_percent(ci.get("low")),
                "CI high": _pct_percent(ci.get("high")),
            }
        )

    return pd.DataFrame(rows)


def _top_missing_df(result: dict[str, Any]) -> pd.DataFrame:
    stdout_json = result.get("stdout_json") or {}
    scenarios = stdout_json.get("scenarios") or []

    rows = []
    for scenario in scenarios:
        for item in scenario.get("top_missing") or []:
            rows.append(
                {
                    "Going": scenario.get("going"),
                    "Missing requirement": item.get("requirement"),
                    "Count": item.get("count"),
                    "% of trials": _pct_percent(item.get("percent_of_trials")),
                    "% of failures": _pct_percent(item.get("percent_of_failures")),
                }
            )

    return pd.DataFrame(rows)


def _starting_hand_draw_cards_from_line(line: str, selected_labels: list[str], goal_mode: str) -> str:
    """
    Infer which goal cards were already naturally available before the listed actions.

    The line CSV currently records the action line, for example:
      Ultra Ball
      Ultra Ball -> Poké Pad
      Fighting Gong

    It does not always say which goal card the action found.

    So:
    - if the line explicitly names one goal card, the other goal card(s) are listed as naturally available.
    - if the line names no goal card, we show the possible naturally available goal cards.
    """
    raw = str(line or "").strip()
    if not raw:
        return "—"

    low = raw.lower()

    clean_goals = [_runtime_goal_name(label) for label in selected_labels]

    mentioned_clean = [
        goal for goal in clean_goals
        if goal.lower() in low
    ]

    natural_clean = [
        goal for goal in clean_goals
        if goal not in mentioned_clean
    ]

    natural_display = [
        _display_label_from_runtime_goal(goal, selected_labels)
        for goal in natural_clean
    ]

    if not natural_display:
        return "—"

    # For an ALL goal, if the action line does not name the found target,
    # this is ambiguous. Example:
    #   Goal: Solrock + Lunatone
    #   Line: Ultra Ball
    #
    # One of Solrock/Lunatone was already naturally available, and Ultra Ball
    # got the other. Since the current line CSV does not identify which one,
    # show the possible list instead of pretending we know.
    if goal_mode == "all" and len(natural_display) > 1:
        return " or ".join(natural_display)

    return ", ".join(natural_display)


def _played_cards_from_line(line: str) -> str:
    raw = str(line or "").strip()
    return raw if raw else "—"


def _success_lines_df(result: dict[str, Any], selected_labels: list[str], goal_mode: str) -> pd.DataFrame:
    path = Path(result["paths"]["lines_csv"])

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # New backend columns. These are actual simulated start/draw contexts,
    # not inferred from the aggregated line.
    if "starting_hand_draw" in df.columns:
        df["Starting Hand + Draw"] = df["starting_hand_draw"].fillna("—").astype(str)
    elif "Starting Hand + Draw" not in df.columns:
        df["Starting Hand + Draw"] = "—"

    if "played" in df.columns:
        df["Played"] = df["played"].fillna("—").astype(str)
    elif "line" in df.columns:
        df["Played"] = df["line"].fillna("—").astype(str)
    else:
        df["Played"] = "—"

    rename_map = {
        "raw_percent": "Raw %",
        "conditional_on_not_natural_percent": "Conditional on not natural %",
        "exact_weighted_percent_of_trials": "Exact weighted % of trials",
    }
    df = df.rename(columns=rename_map)

    preferred = [
        "goal_name",
        "going",
        "Starting Hand + Draw",
        "Played",
        "count",
        "Raw %",
        "Conditional on not natural %",
        "Exact weighted % of trials",
    ]

    existing = [c for c in preferred if c in df.columns]

    hidden = {
        "starting_hand_draw",
        "played",
        "line",
        "Readable line",
        "Readable Line",
    }

    remaining = [c for c in df.columns if c not in existing and c not in hidden]

    return df[existing + remaining]


def _render_runtime_result(result: dict[str, Any], selected_labels: list[str], goal_mode: str) -> None:
    if result.get("returncode") != 0:
        st.error("The goal finder returned a non-zero exit code.")

    st.success(f"Goal finder completed in {result.get('elapsed_seconds', 0):.1f} seconds.")

    summary_df = _scenario_summary_df(result)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No scenario summary was found in the goal-finder output.")

    # Show backend exclusion summary when paths were invalidated.
    stdout_json = result.get("stdout_json") or {}
    scenarios_for_exclusion = stdout_json.get("scenarios") or []

    exclusion_rows = []
    for sc in scenarios_for_exclusion:
        ex = sc.get("played_exclusion_summary") or {}
        if ex.get("enabled"):
            exclusion_rows.append(
                {
                    "Going": sc.get("going"),
                    "Excluded cards": ", ".join(ex.get("excluded_card_names") or []),
                    "Invalidated successes": ex.get("invalidated_successes"),
                }
            )

    if exclusion_rows:
        st.markdown("### Excluded successful paths")
        st.dataframe(pd.DataFrame(exclusion_rows), use_container_width=True, hide_index=True)

    lines_df = _success_lines_df(result, selected_labels, goal_mode)
    if not lines_df.empty:
        st.markdown("### Success lines seen by the simulator")
        st.dataframe(lines_df, use_container_width=True, hide_index=True)

    with st.expander("Command and raw output", expanded=False):
        st.markdown("**Command**")
        st.code(" ".join(result.get("command", [])), language="bash")

        st.markdown("**Output files**")
        st.json(result.get("paths", {}))

        if result.get("stdout"):
            st.markdown("**stdout**")
            st.code(result["stdout"])

        if result.get("stderr"):
            st.markdown("**stderr**")
            st.code(result["stderr"])

        if result.get("report_json") is not None:
            st.markdown("**parsed JSON report**")
            st.json(result["report_json"])


# ---------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# TURN1_FRONTEND_EXCLUDE_HELPERS_V15
# ---------------------------------------------------------------------

def _is_supporter_option(opt: dict[str, Any]) -> bool:
    """
    Best-effort Supporter detection for the deck UI.

    Uses subtype when available, with a fallback list for current test decks.
    Manual exclusion still works even if this detector misses a card.
    """
    text = " ".join(
        str(opt.get(k, ""))
        for k in ["label", "name", "supertype", "subtypes"]
    ).lower()

    if "supporter" in text:
        return True

    known_supporters = {
        "boss's orders",
        "boss’s orders",
        "judge",
        "lillie's determination",
        "lillie’s determination",
        "wally's compassion",
        "wally’s compassion",
        "crispin",
        "professor's research",
        "professor’s research",
        "iono",
        "arven",
        "colress's tenacity",
        "colress’s tenacity",
    }

    clean = _runtime_goal_name(str(opt.get("label") or opt.get("name") or "")).lower()
    return clean in known_supporters


def _ordered_unique_labels(labels: list[str], all_labels: list[str]) -> list[str]:
    seen = set()
    out = []

    order = {label: i for i, label in enumerate(all_labels)}

    for label in sorted(labels, key=lambda x: order.get(x, 10**9)):
        if label not in seen:
            seen.add(label)
            out.append(label)

    return out




# ---------------------------------------------------------------------
# TURN1_FORCE_GOAL_FILE_RUNTIME_V19
# ---------------------------------------------------------------------
# This intentionally redefines _run_goal_finder after earlier versions.
# Python resolves the global function at click time, so this override wins.

def _turn1_v19_goal_min_count(label: str, goal_quantities: dict[str, int] | None) -> int:
    try:
        return max(1, int((goal_quantities or {}).get(label, 1)))
    except Exception:
        return 1


def _turn1_v19_goal_display_name(
    selected_labels: list[str],
    goal_quantities: dict[str, int] | None,
) -> str:
    pieces = []

    for label in selected_labels:
        clean = _runtime_goal_name(label)
        qty = _turn1_v19_goal_min_count(label, goal_quantities)
        pieces.append(clean if qty == 1 else f"{qty}x {clean}")

    return " + ".join(pieces)


def _turn1_v19_write_goal_file(
    path: Path,
    selected_labels: list[str],
    goal_mode: str,
    goal_zone: str,
    goal_quantities: dict[str, int] | None,
) -> None:
    payload = {
        "name": _turn1_v19_goal_display_name(selected_labels, goal_quantities),
        "mode": goal_mode,
        "requirements": [],
    }

    for label in selected_labels:
        clean = _runtime_goal_name(label)
        qty = _turn1_v19_goal_min_count(label, goal_quantities)

        payload["requirements"].append(
            {
                "label": clean if qty == 1 else f"{qty}x {clean}",
                "options": [clean],
                "zone": goal_zone,
                "min_count": qty,
            }
        )

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_goal_finder(
    *,
    all_options: list[dict[str, Any]],
    selected_labels: list[str],
    goal_mode: str,
    goal_zone: str,
    going: str,
    trials: int,
    seed: int,
    max_actions: int,
    example_lines: int,
    workers: int,
    complete_only: bool,
    chain_search: bool,
    goal_quantities: dict[str, int] | None = None,
    excluded_labels: list[str] | None = None,
) -> dict[str, Any]:
    reports_dir = _reports_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_goal = "_".join(_runtime_goal_name(label).split()[0] for label in selected_labels)[:80] or "goal"

    decklist_path = reports_dir / f"deck_{safe_goal}_{stamp}.txt"
    goal_file_path = reports_dir / f"goal_{safe_goal}_{stamp}.json"
    json_path = reports_dir / f"turn1_{safe_goal}_{stamp}.json"
    summary_csv_path = reports_dir / f"turn1_{safe_goal}_{stamp}_summary.csv"
    lines_csv_path = reports_dir / f"turn1_{safe_goal}_{stamp}_lines.csv"
    progress_json_path = reports_dir / f"goal_progress_{stamp}.json"

    _write_decklist_file(all_options, decklist_path)
    _turn1_v19_write_goal_file(
        goal_file_path,
        selected_labels=selected_labels,
        goal_mode=goal_mode,
        goal_zone=goal_zone,
        goal_quantities=goal_quantities or {},
    )

    goal_display_name = _turn1_v19_goal_display_name(selected_labels, goal_quantities or {})

    cmd = [
        sys.executable,
        str(_strict_goal_finder_path()),
        "--compiled",
        str(_compiled_semantics_path()),
        "--decklist",
        str(decklist_path),
        "--goal-file",
        str(goal_file_path),
        "--goal-name",
        goal_display_name,
        "--goal-mode",
        goal_mode,
        "--goal-zone",
        goal_zone,
        "--going",
        going,
        "--trials", str(int(trials)),
        "--workers", str(int(workers)),
        "--seed",
        str(int(seed)),
        "--max-actions",
        str(int(max_actions)),
        "--example-lines",
        str(int(example_lines)),
        "--out",
        str(json_path),
        "--csv-out",
        str(summary_csv_path),
        "--lines-csv",
        str(lines_csv_path),
        "--progress-json",
        str(progress_json_path),
    ]

    if complete_only:
        cmd.append("--complete-only")
    if chain_search:
        cmd.append("--chain-search")

    excluded_labels = excluded_labels or []
    excluded_names = [
        _runtime_goal_name(label)
        for label in excluded_labels
        if str(label).strip()
    ]

    if excluded_names:
        cmd.extend(["--exclude-played", ", ".join(excluded_names)])

    start = time.time()
    proc = _run_goal_finder_subprocess_with_progress(
        cmd,
        cwd=str(_repo_root()),
    )
    elapsed = time.time() - start

    stdout_json = _parse_json_maybe(proc.stdout)

    report_json = None
    if json_path.exists():
        try:
            report_json = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            report_json = None

    return {
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_json": stdout_json,
        "report_json": report_json,
        "paths": {
            "decklist": str(decklist_path),
            "goal_file": str(goal_file_path),
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "lines_csv": str(lines_csv_path),
            "progress_json": str(progress_json_path),
        },
    }


def render_turn1_access_tab(summary=None, card_odds_df=None, deck=None) -> None:
    st.markdown("## Turn 1 Access")
    st.markdown(
        "Pick the cards you want to reach early. This tab shows a stable exact baseline "
        "and can also run the automatic Turn 1 goal finder."
    )

    options = _deck_options(deck, card_odds_df)
    if not options:
        st.info("Analyze a deck first, then open Turn 1 Access.")
        return

    label_to_option = {opt["label"]: opt for opt in options}
    all_labels = list(label_to_option.keys())

    deck_size = _safe_int((summary or {}).get("deck_size", 60), 60) if isinstance(summary, dict) else 60
    basic_count = _safe_int((summary or {}).get("basic_count", 0), 0) if isinstance(summary, dict) else 0

    st.markdown("### Goal setup")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 0.9, 0.9])

        with c1:
            selected_labels = st.multiselect(
                "Which cards are you trying to access?",
                options=all_labels,
                default=_default_goal_labels(options),
                key="turn1_goal_cards_v09",
            )

        with c2:
            goal_mode_label = st.radio(
                "Goal rule",
                options=["Need all selected cards", "Need any one selected card"],
                index=0,
                key="turn1_goal_mode_label_v09",
            )
            goal_mode = "all" if goal_mode_label.startswith("Need all") else "any"

        with c3:
            include_turn_draw = st.toggle(
                "Include natural draw",
                value=True,
                help="Opening 7 plus one natural draw. This is only for the exact baseline.",
                key="turn1_include_draw_v09",
            )

        st.radio(
            "Going first or second",
            options=["Both", "Going first", "Going second"],
            horizontal=True,
            key="turn1_going_baseline_info_v09",
            help="For the exact natural baseline, first and second are the same. The runtime goal finder below handles first/second separately.",
        )

    selected_cards = [label_to_option[label] for label in selected_labels if label in label_to_option]
    if not selected_cards:
        st.info("Choose at least one goal card.")
        return

    # TURN1_GOAL_QUANTITY_CONTROLS_V16
    goal_quantities: dict[str, int] = {}

    with st.container(border=True):
        st.markdown("#### Required copies")

        qty_cols = st.columns(max(1, min(4, len(selected_cards))))

        for idx, card in enumerate(selected_cards):
            with qty_cols[idx % len(qty_cols)]:
                max_copies = max(1, int(card.get("count", 1)))

                goal_quantities[card["label"]] = int(
                    st.number_input(
                        card["label"],
                        min_value=1,
                        max_value=max_copies,
                        value=1,
                        step=1,
                        help=(
                            "Use this when you need more than one copy of the same card. "
                            "Example: choose Basic Fighting Energy once, then set this to 2."
                        ),
                        key=f"turn1_goal_required_copies_{card['key']}",
                    )
                )

        if any(q > 1 for q in goal_quantities.values()):
            st.info(
                "The automatic goal finder will use these required-copy counts. "
                "For example, 2x of a card means the run only succeeds if two physical copies "
                "are in the chosen goal zone."
            )

    # TURN1_ATTACH_REQUIRED_COUNTS_BEFORE_EXACT_V18
    # Required-copy inputs must be attached before exact baseline math,
    # selected-card breakdown, and runtime goal-file generation.
    for card in selected_cards:
        card["required_count"] = _goal_min_count_for_label(card["label"], goal_quantities)

    # TURN1_FORCE_REQUIRED_COUNTS_FOR_RUNTIME_V20
    # These values drive both exact baseline math and the runtime goal file.
    for card in selected_cards:
        card["required_count"] = _goal_min_count_for_label(card["label"], goal_quantities)

    opening_p = _exact_legal_opening_probability(
        deck_size=deck_size,
        basic_count=basic_count,
        selected_cards=selected_cards,
        mode=goal_mode,
        include_turn_draw=False,
    )
    draw_p = _exact_legal_opening_probability(
        deck_size=deck_size,
        basic_count=basic_count,
        selected_cards=selected_cards,
        mode=goal_mode,
        include_turn_draw=True,
    )
    selected_p = draw_p if include_turn_draw else opening_p

    st.markdown("### Exact natural baseline")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Selected result", _pct_fraction(selected_p), "after mulligans")
    with m2:
        st.metric("Opening hand only", _pct_fraction(opening_p), "legal opener")
    with m3:
        st.metric("Opening + natural draw", _pct_fraction(draw_p), "legal opener + 1 card")

    connector = " AND " if goal_mode == "all" else " OR "
    st.success("Goal being checked: " + _goal_display_name_from_quantities(selected_labels, goal_quantities))

    rows = []
    for card in selected_cards:
        one_opening = _exact_legal_opening_probability(
            deck_size=deck_size,
            basic_count=basic_count,
            selected_cards=[card],
            mode="any",
            include_turn_draw=False,
        )
        one_draw = _exact_legal_opening_probability(
            deck_size=deck_size,
            basic_count=basic_count,
            selected_cards=[card],
            mode="any",
            include_turn_draw=True,
        )
        rows.append(
            {
                "Card": card["label"],
                "Copies in deck": card["count"],
                "Required copies": int(card.get("required_count", 1)),
                "Card type": card["supertype"],
                "Basic Pokémon?": card["is_basic"],
                "Opening hand only": _pct_fraction(one_opening),
                "Opening + draw": _pct_fraction(one_draw),
            }
        )

    st.markdown("### Selected card breakdown")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    _render_goal_images(selected_cards)

    # TURN1_REQUIRED_COPY_DEBUG_V20
    st.caption(
        "Runtime goal: "
        + _goal_display_name_from_quantities(selected_labels, {
            card["label"]: int(card.get("required_count", 1))
            for card in selected_cards
        })
    )

    st.markdown("## Automatic goal finder")

    with st.container(border=True):
        r1, r2, r3 = st.columns(3)

        with r1:
            goal_zone = st.selectbox(
                "Goal zone",
                options=["hand_or_in_play", "hand", "in_play", "accessed"],
                index=0,
                key="turn1_goal_zone_v09",
            )
            going = st.selectbox(
                "Going",
                options=["both", "first", "second"],
                index=0,
                key="turn1_runtime_going_v09",
            )

        with r2:
            trials = st.number_input("Trials", min_value=25, max_value=50000, value=5000, step=100, key="turn1_trials_v09")
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=1, step=1, key="turn1_seed_v09")
            example_lines = st.number_input("Example lines", min_value=1, max_value=100, value=25, step=1, key="turn1_example_lines_v09")

        with r3:
            max_actions = st.number_input("Max actions", min_value=1, max_value=30, value=6, step=1, key="turn1_max_actions_v09")
            worker_max = max(1, min(8, os.cpu_count() or 1))
            worker_default = max(1, min(worker_max, int(os.environ.get("TURN1_STREAMLIT_GOAL_WORKERS", "2"))))
            workers = st.number_input(
                "Workers",
                min_value=1,
                max_value=worker_max,
                value=worker_default,
                step=1,
                key="turn1_workers_v01",
                help="Parallel worker processes for Monte Carlo trials. Use 1 if deployment CPU is limited.",
            )
            complete_only = st.toggle("Complete-only", value=True, key="turn1_complete_only_v09")
            chain_search = st.toggle("Chain-search", value=True, key="turn1_chain_search_v09")

        # TURN1_EXCLUDE_PLAYED_UI_V15
        goal_clean_names = {
            _runtime_goal_name(label).lower()
            for label in selected_labels
        }

        supporter_labels = [
            opt["label"]
            for opt in options
            if _is_supporter_option(opt)
            and _runtime_goal_name(opt["label"]).lower() not in goal_clean_names
        ]

        st.markdown("#### Exclude paths")
        st.caption(
            "Use this when you only want success lines that avoid certain cards. "
            "Example: require Raging Bolt + Crispin, but exclude paths that used any other Supporter."
        )

        exclude_other_supporters = st.toggle(
            "Exclude other Supporters except my goal cards",
            value=False,
            key="turn1_exclude_other_supporters_v15",
        )

        manually_excluded = st.multiselect(
            "Exclude paths that play these specific cards",
            options=all_labels,
            default=[],
            help="A run is still allowed to draw these cards. It is only invalid if the simulator plays them in the action line.",
            key="turn1_exclude_played_cards_v15",
        )

        auto_excluded = supporter_labels if exclude_other_supporters else []
        excluded_labels = _ordered_unique_labels(
            list(manually_excluded) + list(auto_excluded),
            all_labels,
        )

        if excluded_labels:
            st.info(
                "Excluded if played: "
                + ", ".join(_runtime_goal_name(label) for label in excluded_labels)
            )

        compiled_path = _compiled_semantics_path()
        if compiled_path.exists():
            st.caption(f"Compiled semantics: {compiled_path}")
        else:
            st.warning(f"Compiled semantics file not found: {compiled_path}")

        run_clicked = st.button("Run automatic goal finder", type="primary", key="turn1_run_goal_finder_v09")

    result_key = "turn1_runtime_result_v09"

    if run_clicked:
        with st.spinner("Running automatic goal finder..."):
            st.session_state[result_key] = _run_goal_finder(
                all_options=options,
                selected_labels=selected_labels,
                goal_mode=goal_mode,
                goal_zone=goal_zone,
                going=going,
                trials=int(trials),
                seed=int(seed),
                max_actions=int(max_actions),
                example_lines=int(example_lines),
                workers=int(workers),
                complete_only=bool(complete_only),
                chain_search=bool(chain_search),
                goal_quantities={
                    card["label"]: int(card.get("required_count", 1))
                    for card in selected_cards
                },
                excluded_labels=excluded_labels,
            )


    result = st.session_state.get(result_key)
    if result:
        if isinstance(result, dict):
            st.caption(
                "Backend trial audit: "
                f"requested={result.get('requested_trials')}, "
                f"actual={result.get('actual_trials')}, "
                f"ok={result.get('trial_count_ok')}, "
                f"workers={result.get('parallel_workers')}"
            )
        _render_runtime_result(result, selected_labels, goal_mode)

    with st.expander("What this tab is doing", expanded=False):
        st.markdown(
            """
            - **Exact natural baseline**: legal opening hand and legal opening hand + natural draw.
            - **Automatic goal finder**: runs the strict Turn 1 simulator and shows first/second results,
              top missing requirements, and success lines.

            The success lines are now shown with a more readable inferred starting-hand note.
            """
        )


# Compatibility aliases
def render_turn1_access_lab(summary=None, card_odds_df=None, deck=None) -> None:
    return render_turn1_access_tab(summary, card_odds_df, deck)


def render_turn1_access_page(summary=None, card_odds_df=None, deck=None) -> None:
    return render_turn1_access_tab(summary, card_odds_df, deck)

# ---------------------------------------------------------------------
# TURN1_DISABLE_GOAL_FINDER_COMMAND_CAPPER
# ---------------------------------------------------------------------
# The old v45 subprocess wrapper capped goal-finder commands to small defaults
# such as 100 trials / 1 worker. That made the UI display one value while the
# backend received another. Keep the timeout wrapper, but stop mutating the
# command arguments.
def _turn1_v45_capped_goal_finder_cmd(cmd):
    return cmd, {}

