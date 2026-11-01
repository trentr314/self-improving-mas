"""
Multi-agent scaffolding: mechanism layer.

This file is the fixed interpreter. Agents never modify it;
they edit protocol.json instead.

Available task phase primitives:
  propose   - each agent generates a proposal
  critique  - each agent critiques others' proposals
  refine    - each agent refines their proposal based on critiques
  vote      - each agent votes on proposals (can't vote for own)
  summarize - a single agent summarizes the round's outcome

Available meta phase primitives:
  meta_propose   - each agent proposes a protocol change
  meta_critique  - each agent critiques others' proposed changes
  meta_refine    - each agent refines their change based on critiques
  meta_vote      - each agent votes on protocol changes
  meta_summarize - summarize the meta-round outcome

meta_propose and meta_vote can't be removed from meta_phases.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

from metrics import (collect_iteration_metrics, save_metrics,
                     print_metrics_summary)
from models import Critique, Proposal, ProtocolChange, Refinement, Vote

load_dotenv()

AVAILABLE_PHASES = ["propose", "critique", "refine", "vote", "summarize"]
AVAILABLE_META_PHASES = ["meta_propose", "meta_critique", "meta_refine", "meta_vote", "meta_summarize"]
LOG_DIR = "logs"
HISTORY_FILE = "evolution_history.json"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def load_history() -> list:
    """Load evolution history from disk."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    """Persist evolution history to disk."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")


def format_history_for_prompt(history: list) -> str:
    """Format evolution history as context for meta-round prompts."""
    if not history:
        return "No prior protocol changes have been made. This is the first iteration."

    lines = ["Previous protocol changes (oldest first):"]
    for entry in history:
        lines.append(
            f"  Iteration {entry['iteration']}: {entry['summary']} "
            f"({entry['field_path']} → {entry['new_value_preview']})"
        )
    return "\n".join(lines)


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
    """Call Gemini with structured output and retry on transient failures."""
    for attempt in range(MAX_RETRIES):
        try:
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
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  (API error: {e}, retrying in {RETRY_DELAY}s...)")
                time.sleep(RETRY_DELAY)
            else:
                raise


# === PHASE HANDLERS ===
# Each takes (client, protocol, agent_names, state, log_file) and returns state,
# which carries data between phases within a round.


def phase_propose(client, protocol, agent_names, state, log_file):
    """Each agent generates a proposal."""
    proposals = []
    system_prompt = protocol.get("roles", {}).get("proposer",
        "You are a creative problem-solver. Propose a concrete, actionable idea. Be specific and concise.")

    for name in agent_names:
        prompt = (
            f"You are agent '{name}'. The current task is:\n\n"
            f"{protocol.get('seed_prompt', 'Propose an improvement.')}\n\n"
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

    if not tally:
        print("  (no valid votes cast)")
        return state

    max_votes = max(tally.values())
    tied = [name for name, count in tally.items() if count == max_votes]
    winner_name = random.choice(tied) if len(tied) > 1 else tied[0]
    if len(tied) > 1:
        print(f"  (tie between {tied}, randomly selected [{winner_name}])")
    winner = next(p for p in proposals if p["agent"] == winner_name)

    log_event(log_file, {"phase": "vote_result", "tally": tally, "winner": winner_name})
    state["winner"] = winner
    return state


def phase_summarize(client, protocol, agent_names, state, log_file):
    """Summarize the round. Works before or after voting."""
    winner = state.get("winner")
    proposals = state.get("proposals", [])
    critiques = state.get("critiques", [])

    if winner:
        # Post-vote: summarize the outcome
        print(f"  Result: [{winner['agent']}] won with: {winner['proposal'].summary}")
        log_event(log_file, {"phase": "summarize", "winner": winner["agent"],
                              "summary": winner["proposal"].summary})
    elif proposals:
        # Pre-vote: summarize proposals and critiques for voters
        system_prompt = protocol.get("roles", {}).get("refiner",
            protocol.get("roles", {}).get("proposer",
                "Summarize the current proposals and how critiques were addressed."))

        proposals_text = "\n\n".join(
            f"[{p['agent']}]: {p['proposal'].summary}\n  {p['proposal'].detail}"
            for p in proposals
        )
        critiques_text = ""
        if critiques:
            critiques_text = "\n\nCritiques received:\n" + "\n".join(
                f"  [{c['critic']}] → [{c['target']}]: {c['critique'].weakness}"
                for c in critiques
            )

        prompt = (
            f"Summarize the current proposals for voters. "
            f"Highlight key differences and how critiques were addressed.\n\n"
            f"Proposals:\n{proposals_text}{critiques_text}"
        )
        summary = call_agent(client, system_prompt, prompt, Proposal)
        print(f"  Summary: {summary.summary}")
        print(f"  {summary.detail}")
        log_event(log_file, {"phase": "summarize", "summary": summary.model_dump()})
        state["round_summary"] = summary
    else:
        print("  (nothing to summarize)")

    return state


PHASE_HANDLERS = {
    "propose": phase_propose,
    "critique": phase_critique,
    "refine": phase_refine,
    "vote": phase_vote,
    "summarize": phase_summarize,
}

CUSTOM_PHASE_MODELS = {
    "Proposal": Proposal,
    "Critique": Critique,
    "Vote": Vote,
    "Refinement": Refinement,
}


def run_custom_phase(client, protocol, agent_names, state, log_file, phase_name):
    """Execute a custom phase defined in protocol.json."""
    custom_def = protocol.get("custom_phases", {}).get(phase_name, {})
    description = custom_def.get("description", f"Execute the '{phase_name}' phase.")
    model_name = custom_def.get("output_model", "Proposal")
    multi_agent = custom_def.get("multi_agent", True)
    schema = CUSTOM_PHASE_MODELS.get(model_name, Proposal)

    context_parts = []
    if state.get("proposals"):
        context_parts.append("Current proposals:")
        for p in state["proposals"]:
            context_parts.append(f"  [{p['agent']}] {p['proposal'].summary}: {p['proposal'].detail}")
    if state.get("winner"):
        w = state["winner"]
        context_parts.append(f"Current winner: [{w['agent']}] {w['proposal'].summary}")
    context = "\n".join(context_parts) if context_parts else "No prior phase data."

    system_prompt = (
        f"You are participating in a multi-agent collaboration phase called '{phase_name}'.\n\n"
        f"Phase instructions: {description}\n\n"
        f"Task context: {protocol.get('seed_prompt', 'Collaborate effectively.')}"
    )

    agents_to_run = agent_names if multi_agent else agent_names[:1]
    results = []

    for name in agents_to_run:
        prompt = (
            f"You are agent '{name}'. Execute the '{phase_name}' phase.\n\n"
            f"Round context:\n{context}\n\n"
            f"Follow the phase instructions and produce your output."
        )
        result = call_agent(client, system_prompt, prompt, schema)
        results.append({"agent": name, "result": result})
        log_event(log_file, {"phase": phase_name, "agent": name,
                              "result": result.model_dump()})

        if hasattr(result, "summary"):
            print(f"  [{name}] {result.summary}")
        elif hasattr(result, "reasoning"):
            print(f"  [{name}] {result.reasoning[:80]}...")
        elif hasattr(result, "weakness"):
            print(f"  [{name}] {result.weakness[:80]}...")

    # Proposal outputs also join the proposal pool so later phases see them
    state[f"custom_{phase_name}"] = results
    if model_name == "Proposal":
        state.setdefault("proposals", []).extend(
            [{"agent": r["agent"], "proposal": r["result"]} for r in results]
        )

    return state


# === META-ROUND ===


def parse_json_value(raw: str):
    """Parse a JSON value from an agent's response, handling common escaping issues."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip outer quotes and unescape, e.g. \"propose\" -> "propose"
    stripped = raw.strip().strip('"').strip("'")
    cleaned = stripped.replace('\\"', '"').replace("\\'", "'")
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Look for a JSON array/object embedded in the string
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

    return raw


def build_meta_system_prompt(protocol, history):
    """Build the system prompt shared across meta-round phases."""
    history_text = format_history_for_prompt(history or [])
    return (
        "You are a systems architect reviewing a multi-agent coordination protocol. "
        "You can change any field in the protocol, including 'meta_phases', which controls "
        "the self-improvement process itself. Be practical: small targeted changes "
        "are better than sweeping rewrites.\n\n"
        f"Available task phase primitives: {json.dumps(AVAILABLE_PHASES)}\n"
        f"Available meta phase primitives: {json.dumps(AVAILABLE_META_PHASES)}\n\n"
        "Agents can also define custom phases (see custom_phases in the protocol).\n"
        "To propose a custom phase, set field_path to 'custom_phases.<phase_name>' and new_value to "
        'a JSON object with: {"description": "...", "output_model": "Proposal|Critique|Vote", '
        '"multi_agent": true|false}\n'
        "Then add the phase name to the 'phases' or 'meta_phases' array.\n\n"
        "IMPORTANT: For the new_value field, provide valid JSON. "
        "For arrays: [\"propose\",\"critique\",\"vote\"] "
        "For strings: just the string without outer quotes. "
        "For numbers: just the number.\n\n"
        f"Evolution history (learn from what was already tried):\n{history_text}"
    )


# === META-PHASE HANDLERS ===
# Mirror the task phase pattern but operate on protocol changes instead of task proposals.
# meta_state carries: meta_proposals, meta_critiques, meta_winner


def meta_phase_propose(client, protocol, agent_names, meta_state, log_file, history):
    """Each agent proposes a change to the protocol."""
    protocol_text = json.dumps(protocol, indent=2)
    system_prompt = build_meta_system_prompt(protocol, history)

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

    meta_state["meta_proposals"] = proposals
    return meta_state


def meta_phase_critique(client, protocol, agent_names, meta_state, log_file, history):
    """Each agent critiques the other agents' proposed protocol changes."""
    proposals = meta_state.get("meta_proposals", [])
    if not proposals:
        print("  (no proposals to critique)")
        return meta_state

    system_prompt = protocol.get("roles", {}).get("critic",
        "You are a sharp but constructive critic. Find weaknesses and suggest fixes. Be concise.")
    critiques = []

    for critic_name in agent_names:
        others = [p for p in proposals if p["agent"] != critic_name]
        if not others:
            continue

        options_text = "\n\n".join(
            f"Change {i}: [{p['agent']}]\n"
            f"  Field: {p['change'].field_path} → {p['change'].new_value}\n"
            f"  Summary: {p['change'].summary}\n"
            f"  Rationale: {p['change'].rationale}"
            for i, p in enumerate(others)
        )
        prompt = (
            f"You are agent '{critic_name}'. Critique the weakest proposed protocol change.\n\n"
            f"{options_text}\n\n"
            f"Pick the change with the biggest weakness (index 0 to {len(others)-1}) "
            f"and explain the weakness and how to fix it."
        )
        critique = call_agent(client, system_prompt, prompt, Critique)
        critique.target_proposal = max(0, min(critique.target_proposal, len(others) - 1))
        target = others[critique.target_proposal]

        critiques.append({"critic": critic_name, "target": target["agent"], "critique": critique})
        log_event(log_file, {"phase": "meta_critique", "critic": critic_name,
                              "target": target["agent"], "critique": critique.model_dump()})
        print(f"  [{critic_name}] → [{target['agent']}]: {critique.weakness[:80]}...")

    meta_state["meta_critiques"] = critiques
    return meta_state


def meta_phase_refine(client, protocol, agent_names, meta_state, log_file, history):
    """Each agent refines their protocol change proposal based on critiques."""
    proposals = meta_state.get("meta_proposals", [])
    critiques = meta_state.get("meta_critiques", [])
    if not proposals or not critiques:
        print("  (nothing to refine)")
        return meta_state

    system_prompt = build_meta_system_prompt(protocol, history)
    refined = []

    for entry in proposals:
        name = entry["agent"]
        change = entry["change"]

        my_critiques = [c for c in critiques if c["target"] == name]
        if not my_critiques:
            refined.append(entry)
            continue

        critique_text = "\n".join(
            f"- [{c['critic']}] Weakness: {c['critique'].weakness}\n"
            f"  Suggested fix: {c['critique'].suggested_fix}"
            for c in my_critiques
        )
        prompt = (
            f"You are agent '{name}'. Here is your original protocol change proposal:\n\n"
            f"  Field: {change.field_path} → {change.new_value}\n"
            f"  Summary: {change.summary}\n"
            f"  Rationale: {change.rationale}\n\n"
            f"You received these critiques:\n{critique_text}\n\n"
            f"Refine your proposal to address the feedback. "
            f"Keep the same field_path or change it if the critique warrants."
        )
        refined_change = call_agent(client, system_prompt, prompt, ProtocolChange)
        refined.append({"agent": name, "change": refined_change})
        log_event(log_file, {"phase": "meta_refine", "agent": name,
                              "change": refined_change.model_dump()})
        print(f"  [{name}] refined: {refined_change.summary}")
        print(f"         {refined_change.field_path} → {refined_change.new_value}")

    meta_state["meta_proposals"] = refined
    return meta_state


def meta_phase_vote(client, protocol, agent_names, meta_state, log_file, history):
    """Each agent votes on protocol changes (can't vote for own)."""
    proposals = meta_state.get("meta_proposals", [])
    if not proposals:
        print("  (no proposals to vote on)")
        return meta_state

    vote_prompt = (
        "You are evaluating proposed changes to a multi-agent coordination protocol. "
        "Vote for the change that would most improve collaboration quality. "
        "Prefer changes that are concrete, testable, and unlikely to break things."
    )
    votes = []
    for voter in agent_names:
        options = [p for p in proposals if p["agent"] != voter]
        if not options:
            continue
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

    if not tally:
        return meta_state

    max_votes = max(tally.values())
    tied = [name for name, count in tally.items() if count == max_votes]
    winner_name = random.choice(tied) if len(tied) > 1 else tied[0]
    if len(tied) > 1:
        print(f"  (tie between {tied}, randomly selected [{winner_name}])")
    winner = next(p for p in proposals if p["agent"] == winner_name)

    meta_state["meta_winner"] = winner
    meta_state["meta_votes"] = votes
    return meta_state


def meta_phase_summarize(client, protocol, agent_names, meta_state, log_file, history):
    """Summarize the meta-round outcome."""
    winner = meta_state.get("meta_winner")
    if winner:
        print(f"  Result: [{winner['agent']}] won with: {winner['change'].summary}")
    else:
        print("  (no winner to summarize)")
    return meta_state


META_PHASE_HANDLERS = {
    "meta_propose": meta_phase_propose,
    "meta_critique": meta_phase_critique,
    "meta_refine": meta_phase_refine,
    "meta_vote": meta_phase_vote,
    "meta_summarize": meta_phase_summarize,
}


def run_custom_meta_phase(client, protocol, agent_names, meta_state, log_file, phase_name):
    """Execute a custom phase within the meta-round, using meta_state keys."""
    custom_def = protocol.get("custom_phases", {}).get(phase_name, {})
    description = custom_def.get("description", f"Execute the '{phase_name}' phase.")
    model_name = custom_def.get("output_model", "Proposal")
    multi_agent = custom_def.get("multi_agent", True)
    schema = CUSTOM_PHASE_MODELS.get(model_name, Proposal)

    # Build context from meta-round state (not task state)
    context_parts = []
    if meta_state.get("meta_proposals"):
        context_parts.append("Current protocol change proposals:")
        for p in meta_state["meta_proposals"]:
            context_parts.append(f"  [{p['agent']}] {p['change'].summary}: {p['change'].rationale}")
    if meta_state.get("meta_winner"):
        w = meta_state["meta_winner"]
        context_parts.append(f"Current winner: [{w['agent']}] {w['change'].summary}")
    context = "\n".join(context_parts) if context_parts else "No prior meta phase data."

    system_prompt = (
        f"You are participating in a meta-round phase called '{phase_name}'.\n\n"
        f"Phase instructions: {description}\n\n"
        f"You are evaluating changes to the coordination protocol itself."
    )

    agents_to_run = agent_names if multi_agent else agent_names[:1]
    results = []

    for name in agents_to_run:
        prompt = (
            f"You are agent '{name}'. Execute the '{phase_name}' phase.\n\n"
            f"Meta-round context:\n{context}\n\n"
            f"Follow the phase instructions and produce your output."
        )
        result = call_agent(client, system_prompt, prompt, schema)
        results.append({"agent": name, "result": result})
        log_event(log_file, {"phase": phase_name, "agent": name,
                              "result": result.model_dump()})

        if hasattr(result, "summary"):
            print(f"  [{name}] {result.summary}")
        elif hasattr(result, "reasoning"):
            print(f"  [{name}] {result.reasoning[:80]}...")
        elif hasattr(result, "weakness"):
            print(f"  [{name}] {result.weakness[:80]}...")

    meta_state[f"custom_{phase_name}"] = results
    return meta_state


def run_meta_round(client, protocol, agent_names, log_file, history=None):
    """Run the meta-round using the data-driven meta_phases from the protocol."""
    meta_phases = protocol.get("meta_phases", ["meta_propose", "meta_vote"])
    meta_state = {}

    for phase_name in meta_phases:
        handler = META_PHASE_HANDLERS.get(phase_name)
        if handler:
            print(f"Phase: {phase_name.upper()}")
            meta_state = handler(client, protocol, agent_names, meta_state, log_file, history)
        elif phase_name in protocol.get("custom_phases", {}):
            print(f"Phase: {phase_name.upper()} (custom)")
            meta_state = run_custom_meta_phase(client, protocol, agent_names, meta_state,
                                                log_file, phase_name)
        else:
            print(f"  (unknown meta phase '{phase_name}', skipping)")
            continue
        print()

    return meta_state.get("meta_winner"), meta_state.get("meta_votes", [])


def apply_protocol_change(protocol: dict, change: ProtocolChange) -> bool:
    """Apply a voted-on change to the protocol. Returns True if applied, False if rejected."""
    value = parse_json_value(change.new_value)

    # The run loop expects these keys to keep their types
    STRUCTURAL_TYPES = {
        "phases": list,
        "meta_phases": list,
        "roles": dict,
        "custom_phases": dict,
    }
    if change.field_path in STRUCTURAL_TYPES:
        expected = STRUCTURAL_TYPES[change.field_path]
        if not isinstance(value, expected):
            print(f"  WARNING: '{change.field_path}' must be a {expected.__name__}, "
                  f"got {type(value).__name__}. Rejecting change.")
            return False

    # Validate custom phase definitions
    if change.field_path.startswith("custom_phases.") and isinstance(value, dict):
        if "description" not in value:
            print("  WARNING: Custom phase missing 'description'. Adding default.")
            value["description"] = f"Execute the {change.field_path.split('.')[-1]} phase."
        if value.get("output_model") not in CUSTOM_PHASE_MODELS:
            value["output_model"] = "Proposal"

    # Drop phase names that aren't built-in or defined in custom_phases
    if change.field_path == "phases":
        custom = set(protocol.get("custom_phases", {}).keys())
        valid = set(AVAILABLE_PHASES) | custom
        value = [p for p in value if isinstance(p, str) and p in valid]
        if not value:
            print("  WARNING: All proposed phases were invalid. Rejecting change.")
            return False

    if change.field_path == "meta_phases":
        custom = set(protocol.get("custom_phases", {}).keys())
        valid = set(AVAILABLE_META_PHASES) | custom
        value = [p for p in value if isinstance(p, str) and p in valid]
        if "meta_vote" not in value or "meta_propose" not in value:
            print("  WARNING: 'meta_phases' must always contain 'meta_propose' and 'meta_vote'. Rejecting change.")
            return False

    if change.field_path == "rounds":
        if not isinstance(value, int) or value < 1:
            print(f"  WARNING: 'rounds' must be a positive integer, got {value!r}. Rejecting change.")
            return False
        value = min(value, 5)  # cap to prevent runaway costs

    try:
        keys = change.field_path.split(".")
        target = protocol
        for key in keys[:-1]:
            if isinstance(target, dict):
                if key not in target:
                    target[key] = {}
                target = target[key]
            else:
                print(f"  WARNING: Cannot traverse into {type(target).__name__} with key '{key}'. Rejecting.")
                return False
        if isinstance(target, dict):
            target[keys[-1]] = value
        else:
            print(f"  WARNING: Cannot set key on {type(target).__name__}. Rejecting.")
            return False
    except Exception as e:
        print(f"  WARNING: Failed to apply change: {e}. Rejecting.")
        return False

    return True


# === MAIN LOOP ===


def run_iteration(client, protocol, agent_names, iteration, log_file, history=None):
    """Run one full iteration: task rounds + meta-round."""
    phases = protocol.get("phases", AVAILABLE_PHASES)
    rounds = protocol.get("rounds", 1)

    meta_phases = protocol.get("meta_phases", ["meta_propose", "meta_vote"])

    print(f"\n{'#'*60}")
    print(f"# ITERATION {iteration}")
    print(f"# Task phases: {phases}")
    print(f"# Meta phases: {meta_phases}")
    print(f"# Rounds: {rounds}")
    print(f"{'#'*60}\n")

    # Task rounds overwrite seed_prompt to chain each winner into the next round.
    # Keep the original so the chained version doesn't get saved to disk.
    original_seed = protocol.get("seed_prompt", "")

    state = {}  # read after the loop, so it has to exist even if rounds == 0
    for round_num in range(rounds):
        print(f"=== Round {round_num + 1} ===\n")
        state = {}

        for phase_name in phases:
            handler = PHASE_HANDLERS.get(phase_name)
            if handler:
                print(f"Phase: {phase_name.upper()}")
                state = handler(client, protocol, agent_names, state, log_file)
            elif phase_name in protocol.get("custom_phases", {}):
                print(f"Phase: {phase_name.upper()} (custom)")
                state = run_custom_phase(client, protocol, agent_names, state,
                                         log_file, phase_name)
            else:
                print(f"  (unknown phase '{phase_name}', skipping)")
                continue
            print()

        winner = state.get("winner")
        if winner:
            print(f"Winner: [{winner['agent']}] {winner['proposal'].summary}")
            print(f"  {winner['proposal'].detail}\n")
            protocol["seed_prompt"] = (
                f"Previous winning proposal: {winner['proposal'].summary}\n"
                f"Detail: {winner['proposal'].detail}\n\n"
                f"Build on this. Propose the next concrete improvement."
            )

    print("=" * 60)
    print("=== META-ROUND ===")
    print(f"=== Meta-phases: {meta_phases}\n")
    meta_winner, meta_votes = run_meta_round(client, protocol, agent_names, log_file,
                                              history=history)

    protocol["seed_prompt"] = original_seed

    if meta_winner is None:
        print("  No valid meta-round winner. Keeping current protocol.\n")
        return protocol

    # Save metrics before applying the change, so they describe the protocol
    # that was actually used this iteration
    state["phases_run"] = phases
    state["num_custom_phases"] = len(protocol.get("custom_phases", {}))
    metrics = collect_iteration_metrics(state, meta_votes,
                                         len(agent_names), iteration)
    save_metrics(metrics)

    print()
    winning_change = meta_winner["change"]
    print(f"Winning change: [{meta_winner['agent']}] {winning_change.summary}")
    print(f"  {winning_change.field_path} → {winning_change.new_value}")
    print(f"  Rationale: {winning_change.rationale}\n")

    applied = apply_protocol_change(protocol, winning_change)

    if applied:
        save_protocol(protocol)
        log_event(log_file, {"phase": "meta_applied", "iteration": iteration,
                              "change": winning_change.model_dump(), "new_protocol": protocol})
    else:
        print("  Change rejected by validation. Keeping current protocol.\n")
        log_event(log_file, {"phase": "meta_rejected", "iteration": iteration,
                              "change": winning_change.model_dump()})

    # Record in evolution history for future iterations
    if history is not None:
        new_value_str = str(winning_change.new_value)
        history.append({
            "iteration": iteration,
            "field_path": winning_change.field_path,
            "new_value_preview": new_value_str[:100],
            "summary": winning_change.summary + (" (REJECTED)" if not applied else ""),
            "rationale": winning_change.rationale,
        })
        save_history(history)

    print("Protocol after this iteration:")
    print(json.dumps(protocol, indent=2))

    return protocol


def main():
    parser = argparse.ArgumentParser(description="Agents that vote on changes to their own protocol")
    parser.add_argument("-n", "--iterations", type=int, default=1,
                        help="Number of iterations to run (default: 1)")
    args = parser.parse_args()

    client = create_client()
    protocol = load_protocol()
    agent_names = [f"Agent_{i}" for i in range(3)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/run_{timestamp}.jsonl"

    os.makedirs(LOG_DIR, exist_ok=True)
    shutil.copy("protocol.json", f"{LOG_DIR}/protocol_v0_{timestamp}.json")

    history = load_history()

    for iteration in range(1, args.iterations + 1):
        protocol = run_iteration(client, protocol, agent_names, iteration, log_file,
                                 history=history)

    print(f"\n{'='*60}")
    print(f"All {args.iterations} iteration(s) complete.")
    print(f"Log: {log_file}")

    print_metrics_summary()


if __name__ == "__main__":
    main()
