# Self-Improving Multi-Agent System

A multi-agent system that evolves its own coordination protocol through recursive self-improvement.

Three AI agents collaborate on tasks using a protocol defined in `protocol.json`. After each task cycle, the agents propose, vote on, and apply changes to the protocol itself — modifying phases, roles, voting rules, and even inventing entirely new phase types. The scaffolding code (`main.py`) remains fixed; only the policy evolves.

## Quick Start

```bash
git clone <repo-url> && cd mas_project
pip install -r requirements.txt
cp .env.example .env
# Add your Google AI API key to .env (get one at https://aistudio.google.com/apikey)
python main.py -n 5
```

Watch as the system evolves from a minimal `propose → vote` loop into something significantly more complex — autonomously.

## How It Works

**Mechanism / Policy Separation**

The core insight: separate the *interpreter* (Python code that runs phases, manages state, enforces rules) from the *policy* (which phases to run, in what order, with what role descriptions). Agents iterate on the policy. The mechanism stays fixed.

```
main.py          ← fixed interpreter, agents never touch this
protocol.json    ← the policy agents evolve via propose/vote meta-rounds
models.py        ← structured output types (Pydantic)
metrics.py       ← per-iteration metrics tracking
```

**The Self-Improvement Loop**

Each iteration:
1. **Task rounds** — agents run through the phases defined in the current protocol
2. **Meta-round** — agents read the protocol, propose one change each, and vote (can't vote for their own proposal)
3. **Apply** — the winning change is applied to `protocol.json`
4. **Repeat** — next iteration uses the updated protocol

**What Agents Can Change**
- Phase sequence (add, remove, reorder from built-in primitives)
- Role descriptions (how proposers, critics, refiners behave)
- Number of rounds per iteration
- Voting rules
- **Custom phases** — agents can define entirely new phase types with natural language descriptions

**What Agents Cannot Change**
- The scaffolding code itself
- The structured output types
- The meta-round governance (propose → vote on protocol changes)

## Observed Evolution

Starting from `["propose", "vote"]` with temperature=0, agents consistently discover:

| Iteration | Change | Protocol State |
|-----------|--------|----------------|
| 1 | Add critique phase | `propose → critique → vote` |
| 2 | Add refine phase | `propose → critique → refine → vote` |
| 3 | Loop critique/refine | `propose → critique → refine → critique → refine → vote` |
| 4 | Add summarize before vote | `... → summarize → vote` |
| 5+ | Role refinements, custom phases | Agents define new phase types like `consolidate_feedback` |

The trajectory is highly reproducible at temperature=0 — different runs produce the same structural evolution.

## Key Design Decisions

- **Enforced governance**: agents can't vote for their own proposals. The scaffolding excludes each agent's proposal from their ballot — no trust required.
- **Structured output**: all agent responses are typed Pydantic models parsed from JSON, not free text. The scaffolding can mechanically validate and route them.
- **Iteration memory**: agents see the history of all prior protocol changes, so they don't repeat experiments.
- **Metrics tracking**: proposal diversity, vote consensus, and critique severity are tracked per-iteration.
- **Rollback**: if a protocol change causes proposal diversity to collapse, the system automatically reverts.

## Architecture

```
┌─────────────────────────────────────────┐
│              main.py (fixed)            │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  │
│  │ propose │→ │ critique │→ │ vote  │  │  ← phase handlers
│  └─────────┘  └──────────┘  └───────┘  │
│        ↑                                │
│   protocol.json (evolving policy)       │
│        ↑                                │
│  ┌──────────────────────────┐           │
│  │  META-ROUND              │           │
│  │  agents propose/vote on  │           │
│  │  protocol changes        │           │
│  └──────────────────────────┘           │
└─────────────────────────────────────────┘
```

## Inspiration

Inspired by the [AI Village](https://theaidigest.org/village) project by AI Digest, which runs multiple AI agents collaboratively but uses minimal coordination scaffolding. This project asks: what if the agents could improve their own coordination?

## Cost

Uses Gemini 2.5 Flash. A 10-iteration run costs roughly $0.10–0.30 depending on phase complexity. Designed to be cheap enough to experiment freely.
