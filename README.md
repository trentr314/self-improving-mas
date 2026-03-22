# Self-Improving Multi-Agent System

A multi-agent system that evolves its own coordination protocol through recursive self-improvement.

Three AI agents collaborate on a task using a protocol defined in `protocol.json`. After each task cycle, the agents propose, vote on, and apply changes to the protocol itself — modifying phases, roles, and even inventing entirely new phase types. The scaffolding code (`main.py`) remains fixed; only the policy evolves.

The system has two parallel evolution tracks: **task phases** (how agents work on the task) and **meta phases** (how agents improve the protocol). Both start minimal and both can grow — agents can modify their own self-improvement process.

## Quick Start

```bash
git clone https://github.com/trentr314/self-improving-mas.git
cd self-improving-mas
python3 -m venv .venv
source .venv/bin/activate
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
1. **Task rounds** — agents run through the `phases` defined in the current protocol (e.g. propose → critique → vote)
2. **Meta-round** — agents run through the `meta_phases` (e.g. meta_propose → meta_critique → meta_vote), proposing and voting on changes to the protocol — including changes to the meta-round itself
3. **Apply** — the winning change is validated and applied to `protocol.json`
4. **Repeat** — next iteration uses the updated protocol

**What Agents Can Change**
- Task phase sequence (add, remove, reorder from built-in primitives)
- Meta phase sequence (same primitives, mirrored for the self-improvement process)
- Role descriptions (how proposers, critics, refiners behave)
- Number of rounds per iteration (capped at 5)
- **Custom phases** — agents can define new phase types with natural language descriptions, usable in both task and meta rounds

**What Agents Cannot Change**
- The scaffolding code itself
- The structured output types
- The requirement that `meta_propose` and `meta_vote` exist in the meta-round (agents can never remove their ability to propose or vote on changes)

## Observed Evolution

Starting from minimal seeds (`phases: ["propose", "vote"]`, `meta_phases: ["meta_propose", "meta_vote"]`) with temperature=0, agents consistently discover:

| Iteration | Change | What Evolved |
|-----------|--------|--------------|
| 1 | Add critique + refine to task phases | `propose → critique → refine → vote` |
| 2 | Add meta_critique + meta_refine to meta phases | Meta-round gains its own deliberation cycle |
| 3 | Add summarize to task phases | `propose → critique → refine → vote → summarize` |
| 4+ | Role refinements, custom phases | Agents define new roles and invent custom phase types |

The structural trajectory is broadly consistent across runs at temperature=0, though not identical — GPU nondeterminism and random tiebreaking can cause variation in later iterations.

## Limitations

Agents bounded by fixed primitives converge quickly (3-5 iterations to exhaust the built-in phases). After that, changes shift to role descriptions and custom phases, which are less structurally impactful. This is a fundamental tension: constrained self-modification converges, unconstrained self-modification (write access to own code) is unsafe and unreliable. This project explores how far you can get within the safe boundary.

## Key Design Decisions

- **Enforced governance**: agents can't vote for their own proposals. The scaffolding excludes each agent's proposal from their ballot — no trust required.
- **Structured output**: all agent responses are typed Pydantic models parsed from JSON, not free text. The scaffolding can mechanically validate and route them.
- **Iteration memory**: agents see the history of all prior protocol changes, so they don't repeat experiments.
- **Metrics tracking**: vote consensus, critique length, and phase counts are tracked per-iteration.
- **Input validation**: protocol changes are validated before applying — invalid phases are filtered, rounds are capped, structural types are enforced, malformed values are rejected rather than crashing the system.
- **Retry on failure**: API calls retry up to 3 times on transient errors to survive long autonomous runs.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                main.py (fixed)                   │
│                                                  │
│  Task Phases              Meta Phases            │
│  ┌─────────┐              ┌──────────────┐      │
│  │ propose │              │ meta_propose │      │
│  │ critique│  ← evolves   │ meta_critique│      │
│  │ refine  │    via        │ meta_refine  │      │
│  │ vote    │    meta       │ meta_vote    │      │
│  │summarize│    rounds     │meta_summarize│      │
│  └─────────┘              └──────┬───────┘      │
│       ↑                          │               │
│       └──── protocol.json ◄──────┘               │
│             (evolving policy)                    │
└──────────────────────────────────────────────────┘
```

Both tracks start minimal and grow independently.
Agents modify task phases through the meta-round, and can modify the meta-round itself — true recursive self-improvement.

## Inspiration

Inspired by the [AI Village](https://theaidigest.org/village) project by AI Digest, which runs multiple AI agents collaboratively but uses minimal coordination scaffolding. This project asks: what if the agents could improve their own coordination?

## Cost

Uses Gemini 2.5 Flash. A 10-iteration run costs roughly $0.10–0.30 depending on phase complexity. Designed to be cheap enough to experiment freely.
