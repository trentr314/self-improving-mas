"""Per-iteration metrics for tracking system evolution."""

import json
import os

METRICS_FILE = "metrics_log.json"


def compute_meta_consensus(meta_votes: list, total_voters: int) -> float:
    """How much do agents agree on protocol changes? 1.0 = unanimous, lower = split."""
    if not meta_votes or total_voters == 0:
        return 0.0
    tally = {}
    for v in meta_votes:
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    max_votes = max(tally.values())
    return max_votes / total_voters


def compute_critique_length(critiques: list) -> float:
    """Average critique length as a proxy for specificity. Normalized to 0-1."""
    if not critiques:
        return 0.0
    lengths = [len(c["critique"].weakness) + len(c["critique"].suggested_fix)
               for c in critiques]
    avg_len = sum(lengths) / len(lengths)
    # Normalize: 50 chars = 0.0, 500+ chars = 1.0
    return min(1.0, max(0.0, (avg_len - 50) / 450))


def collect_iteration_metrics(state: dict, meta_votes: list,
                               num_agents: int, iteration: int) -> dict:
    """Collect all metrics for one iteration."""
    metrics = {"iteration": iteration}
    metrics["vote_consensus"] = compute_meta_consensus(meta_votes, num_agents)
    metrics["critique_length"] = compute_critique_length(
        state.get("critiques", []))
    metrics["num_phases"] = len(state.get("phases_run", []))
    metrics["num_custom_phases"] = state.get("num_custom_phases", 0)
    return metrics


def load_metrics_log() -> list:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return []


def save_metrics(metrics: dict):
    log = load_metrics_log()
    log.append(metrics)
    with open(METRICS_FILE, "w") as f:
        json.dump(log, f, indent=2)
        f.write("\n")


def print_metrics_summary():
    """Print a summary of metrics trends across all iterations."""
    log = load_metrics_log()
    if not log:
        print("No metrics recorded yet.")
        return

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Iter':>4}  {'Vote Consensus':>14}  {'Critique Len':>12}  {'Phases':>6}  {'Custom':>6}")
    print("-" * 50)
    for m in log:
        print(f"{m['iteration']:>4}  "
              f"{m.get('vote_consensus', m.get('meta_consensus', 0)):>14.3f}  "
              f"{m.get('critique_length', m.get('critique_severity', 0)):>12.3f}  "
              f"{m.get('num_phases', 0):>6}  "
              f"{m.get('num_custom_phases', 0):>6}")
    print()
