from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

VERSION = "turn1_runtime_audit_v0_1"


def norm(s: Any) -> str:
    return str(s or "").strip().lower()


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interesting_log_events(log: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = []
    for ev in log or []:
        if not isinstance(ev, dict):
            continue
        blob = norm(json.dumps(ev, ensure_ascii=False))
        event = norm(ev.get("event"))
        if (
            event in {
                "choose_active",
                "use_ability",
                "draw_cards",
                "play_basic_to_bench",
                "last_ditch_catch_found_supporter",
                "ciphermaniac_put_target_on_top",
                "shuffle_deck",
                "strict_runtime_block",
                "discard_fodder_for_cost",
                "attach_energy",
            }
            or "run errand" in blob
            or "teal dance" in blob
            or "ciphermaniac" in blob
            or "last-ditch" in blob
            or "unfair stamp" in blob
            or "flip the script" in blob
        ):
            keep.append(ev)
    return keep


def run_errand_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    line = str(result.get("line") or "")
    log = result.get("log") or []
    has_run_errand_line = "Run Errand" in line
    use_events = [ev for ev in log if isinstance(ev, dict) and ev.get("event") == "use_ability" and ev.get("ability") == "Run Errand"]
    draw_events = [ev for ev in log if isinstance(ev, dict) and ev.get("event") == "draw_cards" and "run_errand" in norm(ev.get("stage"))]
    cipher_before = any(isinstance(ev, dict) and "ciphermaniac" in norm(json.dumps(ev, ensure_ascii=False)) for ev in log)
    return {
        "line_has_run_errand": has_run_errand_line,
        "active": result.get("active"),
        "active_is_mega_kangaskhan_ex": norm(result.get("active")) == norm("Mega Kangaskhan ex"),
        "run_errand_use_events": len(use_events),
        "run_errand_draw_events": len(draw_events),
        "run_errand_drawn_cards": [ev.get("drawn") for ev in draw_events],
        "ciphermaniac_seen_before_or_in_line": cipher_before or "Ciphermaniac" in line,
        "looks_proven_legal": (not has_run_errand_line) or (norm(result.get("active")) == norm("Mega Kangaskhan ex") and len(use_events) >= 1 and len(draw_events) >= 1),
    }


def teal_dance_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    line = str(result.get("line") or "")
    log = result.get("log") or []
    has_td = "Teal Dance" in line
    use_events = [ev for ev in log if isinstance(ev, dict) and ev.get("event") == "use_ability" and ev.get("ability") == "Teal Dance"]
    draw_events = [ev for ev in log if isinstance(ev, dict) and ev.get("event") == "draw_cards" and "teal_dance" in norm(ev.get("stage"))]
    attached = [ev.get("attached") for ev in use_events if ev.get("attached")]
    return {
        "line_has_teal_dance": has_td,
        "teal_dance_use_events": len(use_events),
        "attached_cards": attached,
        "teal_dance_draw_events": len(draw_events),
        "teal_dance_drawn_cards": [ev.get("drawn") for ev in draw_events],
        "looks_proven_legal": (not has_td) or (len(use_events) >= 1 and bool(attached)),
    }


def scenario_examples(sc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(sc.get("example_successes") or []) + list(sc.get("example_failures") or [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Turn-1 goal-finder report logs for runtime legality/provenance.")
    ap.add_argument("--report", required=True, help="JSON report produced by run_turn1_goal_finder_strict.py")
    ap.add_argument("--line-contains", default="Run Errand", help="Only print detailed examples whose line contains this text. Use empty string for all examples.")
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--show-log", action="store_true", help="Print filtered action log for matching examples.")
    args = ap.parse_args()

    data = load_json(args.report)
    print(json.dumps({
        "version": VERSION,
        "report": args.report,
        "goal_name": data.get("goal_name"),
        "goal_mode": data.get("goal_mode"),
        "goal_zone": data.get("goal_zone"),
        "trials": data.get("trials"),
    }, indent=2, ensure_ascii=False))

    for sc in data.get("scenarios", []) or []:
        going = sc.get("going")
        sm = sc.get("summary") or {}
        top_lines = sm.get("top_success_lines") or []
        flags = {
            "contains_unfair_stamp_top_lines": any("Unfair Stamp" in str(r.get("line")) for r in top_lines),
            "contains_flip_the_script_top_lines": any("Flip the Script" in str(r.get("line")) for r in top_lines),
            "contains_run_errand_top_lines": any("Run Errand" in str(r.get("line")) for r in top_lines),
        }
        print("\n=== Scenario", going, "===")
        print(json.dumps({
            "raw_sim_percent": sm.get("percent"),
            "final_exact_plus_sim_percent": (sc.get("exact_plus_simulation") or {}).get("final_exact_plus_sim_percent"),
            "success_by_stage": sm.get("success_by_stage"),
            "flags": flags,
            "top_lines": top_lines[:15],
        }, indent=2, ensure_ascii=False))

        examples = scenario_examples(sc)
        needle = args.line_contains
        if needle:
            examples = [r for r in examples if needle in str(r.get("line") or "")]
        selected = examples[: args.max_examples]
        print(f"\nDetailed examples matching {needle!r}: {len(selected)} shown / {len(examples)} available")
        for i, r in enumerate(selected, start=1):
            summary = {
                "example": i,
                "success": r.get("success"),
                "success_stage": r.get("success_stage"),
                "line": r.get("line"),
                "active": r.get("active"),
                "actions_used": r.get("actions_used"),
                "final_hand_size": r.get("final_hand_size"),
                "final_deck_size": r.get("final_deck_size"),
                "accessed_goal_piece_names": r.get("accessed_goal_piece_names"),
                "run_errand_checks": run_errand_checks(r),
                "teal_dance_checks": teal_dance_checks(r),
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            if args.show_log:
                print("filtered_log:")
                print(json.dumps(interesting_log_events(r.get("log") or []), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
