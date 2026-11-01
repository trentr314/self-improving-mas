# self-improving-mas

A small experiment I built for fun. Three LLM agents work on a toy task (designing a text adventure game), and after each round they propose and vote on changes to the rules they work under. The Python code stays fixed. The only thing that changes is `protocol.json`, which lists the steps to run and describes each role. I mostly wanted to see what they'd do with that.

The name oversells it a bit. Nothing here checks whether the agents' actual work gets better, so "self-modifying" would be more accurate than "self-improving." The three agents are also the same model with the same prompt at temperature 0; they differ only in their names and in which proposals they see, since nobody critiques or votes on their own.

## Running it

```bash
git clone https://github.com/trentr314/self-improving-mas.git
cd self-improving-mas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add a Google AI API key (https://aistudio.google.com/apikey)
python main.py -n 5     # run 5 iterations
```

A run rewrites `protocol.json` in place (the starting version is copied into `logs/` first) and appends to `evolution_history.json`, which agents see in later runs. To start over, restore `protocol.json` from git and delete `evolution_history.json` and `metrics_log.json`.

## How it works

Each iteration has two parts:

1. A task round. The agents run the steps listed under `phases` in `protocol.json`. This starts as just `propose`, `vote`.
2. A meta-round. They run the steps under `meta_phases` (starting as `meta_propose`, `meta_vote`), except now each agent proposes one edit to `protocol.json`. The winning edit gets checked and applied, and the next iteration uses the new protocol.

Since `meta_phases` lives in `protocol.json` too, the agents can also change how the meta-round itself works. That's the "recursive" part, and it's about as far as the recursion goes.

The built-in steps are `propose`, `critique`, `refine`, `vote`, and `summarize`, plus a `meta_` version of each. Besides reordering those, agents can rewrite the role descriptions, set `rounds` (capped at 5), or define a custom step by writing a short description of it, which gets passed to the model as instructions when the step runs.

They can't edit the Python, change the response schemas, or remove `meta_propose` or `meta_vote` (so they can't vote away their own ability to vote). They also can't vote for their own proposal; the code just leaves it off their ballot.

```
main.py          runs the phases and applies the winning change (agents never touch this)
protocol.json    the config the agents edit
models.py        Pydantic schemas for every agent response
metrics.py       a few per-iteration numbers: vote agreement, critique length, phase count
```

## What happened

I only did a handful of runs, so this is anecdote. Starting from the minimal protocol at temperature 0:

- Iteration 1: all four runs from the minimal protocol made the same first move, adding `critique` and `refine` to the task phases.
- Iteration 2: in the one run I let go longer, they gave the meta-round the same treatment (`meta_critique`, `meta_refine`).
- Iteration 3: they added `summarize` to the end of the task phases.
- Iterations 4 and 5 (continued in a second run): they defined a custom step called `meta_summarize_protocol_changes`, then defined it again the next iteration. Neither time did anyone add it to `meta_phases`, so it never ran.

So they rebuilt a pretty standard propose/critique/revise/vote loop and then started fiddling. Once the built-in steps are used up there isn't much left to change except wording and custom steps, which is probably the main limit of keeping them inside a fixed set of primitives. Letting them edit their own code would get around that, but it also seemed like a good way to end up with a broken script.

Each run writes a JSONL log to `logs/`, one line per agent action, if you want to see the actual proposals and votes.

## Notes

- Inspired by [AI Village](https://theaidigest.org/village), where several AI agents work together with fairly light structure. I wondered what would happen if they got to set up the structure themselves.
- Uses Gemini 2.5 Flash through the `google-genai` SDK. No agent frameworks.
- An iteration makes 12 API calls with the starting protocol and up to about 24 once all the built-in steps are turned on (more if `rounds` goes up), so it's cheap to play with on Flash.
