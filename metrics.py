"""Per-iteration metrics for tracking system evolution."""

import json
import os
from difflib import SequenceMatcher

METRICS_FILE = "metrics_log.json"


def text_similarity(a: str, b: str) -> float:
    """Simple string similarity ratio between two texts."""
    return SequenceMatcher(None, a, b).ratio()


def compute_proposal_diversity(proposals: list) -> float:
    """Average pairwise dissimilarity between proposal texts. 1.0 = maximally diverse."""
    if len(proposals) < 2:
        return 0.0
    texts = [f"{p['proposal'].summary} {p['proposal'].detail}" for p in proposals]
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities.append(text_similarity(texts[i], texts[j]))
    return 1.0 - (sum(similarities) / len(similarities))


def compute_vote_unanimity(votes: list, total_voters: int) -> float:
    """How concentrated are votes? 1.0 = all voted the same, 0.0 = evenly split."""
    if not votes or total_voters == 0:
        return 0.0
    tally = {}
    for v in votes:
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    max_votes = max(tally.values())
    return max_votes / total_voters


def compute_critique_severity(critiques: list) -> float:
    """Average critique length as a proxy for specificity. Normalized to 0-1."""
    if not critiques:
        return 0.0
    lengths = [len(c["critique"].weakness) + len(c["critique"].suggested_fix)
               for c in critiques]
    avg_len = sum(lengths) / len(lengths)
    # Normalize: 50 chars = 0.0, 500+ chars = 1.0
    return min(1.0, max(0.0, (avg_len - 50) / 450))


def compute_meta_consensus(meta_votes: list, total_voters: int) -> float:
    """How much do agents agree on protocol changes? Same as vote unanimity."""
    if not meta_votes or total_voters == 0:
        return 0.0
    tally = {}
    for v in meta_votes:
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    max_votes = max(tally.values())
    return max_votes / total_voters


def collect_iteration_metrics(state: dict, meta_votes: list,
                               num_agents: int, iteration: int) -> dict:
    """Collect all metrics for one iteration."""
    metrics = {"iteration": iteration}
    metrics["proposal_diversity"] = compute_proposal_diversity(
        state.get("proposals", []))
    # We don't have direct access to task votes here, so we track meta consensus
    metrics["meta_consensus"] = compute_meta_consensus(meta_votes, num_agents)
    metrics["critique_severity"] = compute_critique_severity(
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
    print(f"{'Iter':>4}  {'Diversity':>9}  {'Consensus':>9}  {'Crit.Sev':>8}  {'Phases':>6}  {'Custom':>6}")
    print("-" * 56)
    for m in log:
        print(f"{m['iteration']:>4}  "
              f"{m['proposal_diversity']:>9.3f}  "
              f"{m['meta_consensus']:>9.3f}  "
              f"{m['critique_severity']:>8.3f}  "
              f"{m.get('num_phases', 0):>6}  "
              f"{m.get('num_custom_phases', 0):>6}")
    print()
