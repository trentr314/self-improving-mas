"""
Multi-agent scaffolding: mechanism layer.

This file is the FIXED interpreter. Agents never modify it.
They iterate on protocol.json instead.

Available phase primitives (agents can reorder/add/remove these):
  propose   - each agent generates a proposal
  critique  - each agent critiques others' proposals
  refine    - each agent refines their proposal based on critiques
  vote      - each agent votes on proposals (can't vote for own)
  summarize - a single agent summarizes the round's outcome
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

from models import Critique, Proposal, ProtocolChange, Refinement, Vote

load_dotenv()

AVAILABLE_PHASES = ["propose", "critique", "refine", "vote", "summarize"]
LOG_DIR = "logs"


def create_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-key-here":
        print("Set your GOOGLE_API_KEY in .env first.")
        print("Get one at: https://aistudio.google.com/apikey")
        raise SystemExit(1)
    return genai.Client(api_key=api_key)


def load_protocol(path="protocol.json"):
    with open(path) as f:
        return json.load(f)


def save_protocol(protocol: dict, path="protocol.json"):
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")


def log_event(log_file: str, event: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")


def call_agent(client, system_prompt: str, user_prompt: str, schema):
    """Call Gemini with structured output, returning a parsed Pydantic model."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    )
    return schema.model_validate_json(response.text)


# === PHASE HANDLERS ===
# Each takes (client, protocol, agent_names, round_state) and returns updated round_state.
# round_state carries data between phases within a round.


def phase_propose(client, protocol, agent_names, state, log_file):
    """Each agent generates a proposal."""
    proposals = []
    system_prompt = protocol.get("roles", {}).get("proposer",
        "You are a creative problem-solver. Propose a concrete, actionable idea. Be specific and concise.")

    for name in agent_names:
        prompt = (
            f"You are agent '{name}'. The current task is:\n\n"
            f"{protocol['seed_prompt']}\n\n"
            "Propose one concrete improvement or design idea."
        )
        proposal = call_agent(client, system_prompt, prompt, Proposal)
        proposals.append({"agent": name, "proposal": proposal})
        log_event(log_file, {"phase": "propose", "agent": name, "proposal": proposal.model_dump()})
        print(f"  [{name}] {proposal.summary}")

    state["proposals"] = proposals
    return state


def phase_critique(client, protocol, agent_names, state, log_file):
    """Each agent critiques the other agents' proposals."""
    proposals = state.get("proposals", [])
    if not proposals:
        print("  (no proposals to critique)")
        return state

    system_prompt = protocol.get("roles", {}).get("critic",
        "You are a sharp but constructive critic. Find weaknesses and suggest fixes. Be concise.")
    critiques = []

    for critic_name in agent_names:
        others = [p for p in proposals if p["agent"] != critic_name]
        if not others:
            continue

        proposals_text = "\n\n".join(
            f"Proposal {i}: [{p['agent']}]\n"
            f"  Summary: {p['proposal'].summary}\n"
            f"  Detail: {p['proposal'].detail}"
            for i, p in enumerate(others)
        )
        prompt = (
            f"You are agent '{critic_name}'. Critique the weakest proposal.\n\n"
            f"{proposals_text}\n\n"
            f"Pick the proposal with the biggest weakness (index 0 to {len(others)-1}) "
            f"and explain the weakness and how to fix it."
        )
        critique = call_agent(client, system_prompt, prompt, Critique)
        critique.target_proposal = max(0, min(critique.target_proposal, len(others) - 1))
        target = others[critique.target_proposal]

        critiques.append({"critic": critic_name, "target": target["agent"], "critique": critique})
        log_event(log_file, {"phase": "critique", "critic": critic_name,
                              "target": target["agent"], "critique": critique.model_dump()})
        print(f"  [{critic_name}] → [{target['agent']}]: {critique.weakness[:80]}...")

    state["critiques"] = critiques
    return state


def phase_refine(client, protocol, agent_names, state, log_file):
    """Each agent refines their proposal based on critiques received."""
    proposals = state.get("proposals", [])
    critiques = state.get("critiques", [])
    if not proposals or not critiques:
        print("  (nothing to refine)")
        return state

    system_prompt = protocol.get("roles", {}).get("proposer",
        "You are a creative problem-solver. Refine your proposal based on feedback.")
    refined = []

    for i, entry in enumerate(proposals):
        name = entry["agent"]
        proposal = entry["proposal"]

        # Gather critiques aimed at this agent
        my_critiques = [c for c in critiques if c["target"] == name]
        if not my_critiques:
            refined.append(entry)  # no critiques, keep original
            continue

        critique_text = "\n".join(
            f"- [{c['critic']}] Weakness: {c['critique'].weakness}\n"
            f"  Suggested fix: {c['critique'].suggested_fix}"
            for c in my_critiques
        )
        prompt = (
            f"You are agent '{name}'. Here is your original proposal:\n\n"
            f"  Summary: {proposal.summary}\n"
            f"  Detail: {proposal.detail}\n\n"
            f"You received these critiques:\n{critique_text}\n\n"
            f"Refine your proposal to address the feedback. "
            f"Set original_proposal to {i}."
        )
        refinement = call_agent(client, system_prompt, prompt, Refinement)

        # Replace the proposal with the refined version
        refined_proposal = Proposal(
            summary=refinement.summary,
            detail=refinement.detail,
            rationale=f"Refined: {refinement.changes_made}",
        )
        refined.append({"agent": name, "proposal": refined_proposal})
        log_event(log_file, {"phase": "refine", "agent": name,
                              "refinement": refinement.model_dump()})
        print(f"  [{name}] refined: {refinement.summary}")

    state["proposals"] = refined
    return state


def phase_vote(client, protocol, agent_names, state, log_file):
    """Each agent votes on proposals (excluding their own). Sets state['winner']."""
    proposals = state.get("proposals", [])
    if not proposals:
        print("  (nothing to vote on)")
        return state

    system_prompt = protocol.get("roles", {}).get("critic",
        "You are evaluating proposals. Vote for the best one.")
    votes = []

    for voter in agent_names:
        options = [p for p in proposals if p["agent"] != voter]
        if not options:
            continue

        options_text = "\n\n".join(
            f"Proposal {i}: [{p['agent']}]\n"
            f"  Summary: {p['proposal'].summary}\n"
            f"  Detail: {p['proposal'].detail}\n"
            f"  Rationale: {p['proposal'].rationale}"
            for i, p in enumerate(options)
        )
        prompt = (
            f"You are agent '{voter}'. Vote for the best proposal.\n\n"
            f"Options:\n{options_text}\n\n"
            f"Vote by index (0 to {len(options)-1})."
        )
        vote = call_agent(client, system_prompt, prompt, Vote)
        vote.chosen_proposal = max(0, min(vote.chosen_proposal, len(options) - 1))
        chosen = options[vote.chosen_proposal]

        votes.append({"voter": voter, "voted_for": chosen["agent"], "reasoning": vote.reasoning})
        log_event(log_file, {"phase": "vote", "voter": voter, "voted_for": chosen["agent"],
                              "reasoning": vote.reasoning})
        print(f"  [{voter}] → [{chosen['agent']}]: {vote.reasoning[:80]}...")

    tally = {}
    for v in votes:
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    winner_name = max(tally, key=tally.get)
    winner = next(p for p in proposals if p["agent"] == winner_name)

    log_event(log_file, {"phase": "vote_result", "tally": tally, "winner": winner_name})
    state["winner"] = winner
    return state


def phase_summarize(client, protocol, agent_names, state, log_file):
    """One agent summarizes the round outcome."""
    winner = state.get("winner")
    if not winner:
        print("  (no winner to summarize)")
        return state

    print(f"  Summary: [{winner['agent']}] won with: {winner['proposal'].summary}")
    log_event(log_file, {"phase": "summarize", "winner": winner["agent"],
                          "summary": winner["proposal"].summary})
    return state


PHASE_HANDLERS = {
    "propose": phase_propose,
    "critique": phase_critique,
    "refine": phase_refine,
    "vote": phase_vote,
    "summarize": phase_summarize,
}


# === META-ROUND ===


def parse_json_value(raw: str):
    """Parse a JSON value from an agent's response, handling common escaping issues."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try stripping outer quotes and unescaping (handles \"propose\" → "propose")
    stripped = raw.strip().strip('"').strip("'")
    # Fix escaped quotes
    cleaned = stripped.replace('\\"', '"').replace("\\'", "'")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find a JSON structure inside the string
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

    # Give up, return as string
    return raw


def meta_round(client, protocol, agent_names, log_file):
    """Agents propose and vote on changes to the protocol itself."""
    protocol_text = json.dumps(protocol, indent=2)

    system_prompt = (
        "You are a systems architect reviewing a multi-agent coordination protocol. "
        "Propose one specific, concrete change to improve how agents collaborate. "
        "You can change any field in the protocol. Be practical — small targeted changes "
        "are better than sweeping rewrites.\n\n"
        f"Available phase primitives: {json.dumps(AVAILABLE_PHASES)}\n"
        "Agents can only execute these phases — don't invent new phase names.\n\n"
        "IMPORTANT: For the new_value field, provide valid JSON. "
        "For arrays: [\"propose\",\"critique\",\"vote\"] "
        "For strings: just the string without outer quotes. "
        "For numbers: just the number."
    )

    # Propose
    proposals = []
    for name in agent_names:
        prompt = (
            f"You are agent '{name}'. Here is the current protocol:\n\n"
            f"```json\n{protocol_text}\n```\n\n"
            "Propose one change to this protocol that would improve the quality "
            "of multi-agent collaboration. Specify the exact field path and new value."
        )
        change = call_agent(client, system_prompt, prompt, ProtocolChange)
        proposals.append({"agent": name, "change": change})
        log_event(log_file, {"phase": "meta_propose", "agent": name, "change": change.model_dump()})
        print(f"  [{name}] {change.summary}")
        print(f"         {change.field_path} → {change.new_value}")

    # Vote
    print()
    print("Phase: META-VOTE")
    vote_prompt = (
        "You are evaluating proposed changes to a multi-agent coordination protocol. "
        "Vote for the change that would most improve collaboration quality. "
        "Prefer changes that are concrete, testable, and unlikely to break things."
    )
    votes = []
    for voter in agent_names:
        options = [p for p in proposals if p["agent"] != voter]
        options_text = "\n\n".join(
            f"Change {i}: [{p['agent']}]\n"
            f"  Field: {p['change'].field_path} → {p['change'].new_value}\n"
            f"  Summary: {p['change'].summary}\n"
            f"  Rationale: {p['change'].rationale}"
            for i, p in enumerate(options)
        )
        prompt = (
            f"You are agent '{voter}'. Vote for the best protocol change.\n\n"
            f"Options:\n{options_text}\n\n"
            f"Vote by index (0 to {len(options)-1})."
        )
        vote = call_agent(client, vote_prompt, prompt, Vote)
        vote.chosen_proposal = max(0, min(vote.chosen_proposal, len(options) - 1))
        chosen = options[vote.chosen_proposal]
        votes.append({"voter": voter, "voted_for": chosen["agent"]})
        log_event(log_file, {"phase": "meta_vote", "voter": voter,
                              "voted_for": chosen["agent"], "reasoning": vote.reasoning})
        print(f"  [{voter}] → [{chosen['agent']}]: {vote.reasoning[:80]}...")

    tally = {}
    for v in votes:
        tally[v["voted_for"]] = tally.get(v["voted_for"], 0) + 1
    winner_name = max(tally, key=tally.get)
    winner = next(p for p in proposals if p["agent"] == winner_name)

    return winner


def apply_protocol_change(protocol: dict, change: ProtocolChange) -> dict:
    """Apply a voted-on change to the protocol. Handles JSON parsing robustly."""
    value = parse_json_value(change.new_value)

    # Validate phase lists — only allow known primitives
    if change.field_path == "phases" and isinstance(value, list):
        value = [p for p in value if p in AVAILABLE_PHASES]
        if not value:
            print("  WARNING: All proposed phases were invalid. Keeping original.")
            return protocol

    keys = change.field_path.split(".")
    target = protocol
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value

    return protocol


# === MAIN LOOP ===


def run_iteration(client, protocol, agent_names, iteration, log_file):
    """Run one full iteration: task rounds + meta-round."""
    print(f"\n{'#'*60}")
    print(f"# ITERATION {iteration}")
    print(f"# Phases: {protocol['phases']}")
    print(f"# Rounds: {protocol['rounds']}")
    print(f"{'#'*60}\n")

    # Task rounds
    for round_num in range(protocol["rounds"]):
        print(f"=== Round {round_num + 1} ===\n")
        state = {}

        for phase_name in protocol["phases"]:
            handler = PHASE_HANDLERS.get(phase_name)
            if not handler:
                print(f"  (unknown phase '{phase_name}', skipping)")
                continue
            print(f"Phase: {phase_name.upper()}")
            state = handler(client, protocol, agent_names, state, log_file)
            print()

        winner = state.get("winner")
        if winner:
            print(f"Winner: [{winner['agent']}] — {winner['proposal'].summary}")
            print(f"  {winner['proposal'].detail}\n")
            protocol["seed_prompt"] = (
                f"Previous winning proposal: {winner['proposal'].summary}\n"
                f"Detail: {winner['proposal'].detail}\n\n"
                f"Build on this. Propose the next concrete improvement."
            )

    # Meta-round
    print("=" * 60)
    print("=== META-ROUND: Protocol Self-Improvement ===\n")
    print("Phase: META-PROPOSE")
    meta_winner = meta_round(client, protocol, agent_names, log_file)
    print()

    winning_change = meta_winner["change"]
    print(f"Winning change: [{meta_winner['agent']}] — {winning_change.summary}")
    print(f"  {winning_change.field_path} → {winning_change.new_value}")
    print(f"  Rationale: {winning_change.rationale}\n")

    # Apply to a clean reload (don't persist mutated seed_prompt)
    clean_protocol = load_protocol()
    apply_protocol_change(clean_protocol, winning_change)
    save_protocol(clean_protocol)

    log_event(log_file, {"phase": "meta_applied", "iteration": iteration,
                          "change": winning_change.model_dump(), "new_protocol": clean_protocol})

    print("Protocol updated. New protocol:")
    print(json.dumps(clean_protocol, indent=2))

    return clean_protocol


def main():
    parser = argparse.ArgumentParser(description="Self-improving multi-agent system")
    parser.add_argument("-n", "--iterations", type=int, default=1,
                        help="Number of self-improvement iterations to run (default: 1)")
    args = parser.parse_args()

    client = create_client()
    protocol = load_protocol()
    agent_names = [f"Agent_{i}" for i in range(3)]

    # Each run gets its own log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/run_{timestamp}.jsonl"

    # Back up the original protocol
    shutil.copy("protocol.json", f"{LOG_DIR}/protocol_v0_{timestamp}.json")

    for iteration in range(1, args.iterations + 1):
        protocol = run_iteration(client, protocol, agent_names, iteration, log_file)

    print(f"\n{'='*60}")
    print(f"All {args.iterations} iteration(s) complete.")
    print(f"Log: {log_file}")
    print(f"Protocol evolved through {args.iterations} self-improvement cycle(s).")


if __name__ == "__main__":
    main()
