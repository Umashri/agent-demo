# agent-demo

A minimal tool-calling AI agent, built from scratch to understand how agents actually work — not to ship a product.

I deliberately used **no agent frameworks** (no LangChain, no LlamaIndex, no agent libraries). The agent loop, the tool-calling, and the safety limits are all written by hand against the Gemini API, so none of the mechanism is hidden. The goal was understanding, not convenience.

## What it does

The agent is given a set of tools (plain Python functions) and a question. The model then decides, on its own, whether to call a tool, reads the result, and decides again — looping until it produces a final text answer. It:

- Picks the right tool for a question (a calculator vs. a planet-fact lookup).
- Chains multiple tools for one question (e.g. compute a number *and* look up a fact, then combine them).
- Answers directly, with no tool, when none is needed (e.g. a greeting).

## How an agent works (the core idea)

An agent is **an LLM in a loop with tools**. Three pieces:

1. **Tools** — ordinary functions. Nothing AI about them; they're just capabilities.
2. **A model that knows the tools exist** — the library passes each function's name, parameters, and docstring to the model, so it knows what it can call and when.
3. **The loop** — each turn, the model either returns a final answer (done) or requests a tool call. The code runs the requested tool, sends the result back, and the model decides its next step. This repeats — a structured conversation in which some turns are the model talking to the tools — until a final answer.

## Failure modes I hit, and how I handled them

The interesting part of building an agent is not the happy path — it's what breaks.

### 1. Infinite loop

On the first run, the agent called the same function with the same arguments over and over, never finishing. The trigger was a bug that stopped the loop from advancing to the model's new response — so it kept re-reading the original "call this tool" instruction forever.

Rather than rely on never writing that bug again, I added two independent safeguards, because they catch different failures:

- **Loop detection** — every tool call is recorded as a signature (`name(args)`). If the same call repeats 3 times, the agent aborts. This catches an agent *spinning in place* on the same action.
- **Step cap** — a hard limit on the number of tool-calling rounds. This catches an agent that *wanders* through many different but unproductive steps without ever repeating exactly.

Loop detection catches exact repetition; the step cap catches non-repeating runaway. Production agent systems layer both for exactly this reason.

### 2. Tool-capability mismatch

Asked "the square root of 144 plus Jupiter's moons", the agent reasoned correctly — it fetched the moon count, then called the calculator with `sqrt(144) + 95`. But the calculator crashed: it used Python's `eval` without math functions available, so `sqrt` was undefined, and the crash took down the whole agent.

The key insight: **the agent did everything right — the tool was too weak for what the agent reasonably asked of it.** The model saw a tool named "calculator" and sensibly assumed it could handle `sqrt`. The real lesson is that tools must be robust to how the model *actually* calls them, not just how the author imagined.

Fixes:
- The calculator now supports common math functions.
- It runs `eval` with built-ins stripped out (only math functions exposed) — safer than raw `eval`, though still not production-grade.
- Most importantly, it catches its own errors and **returns an error message as data** instead of crashing. The agent then receives the error as a normal tool result and can reason about it and recover, rather than the whole process dying.

## Stack

- Python
- Google Gemini API (`gemini-3.6-flash`), function-calling
- No frameworks, by design

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install google-generativeai python-dotenv
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

Then:

```bash
python3 agent.py
```

Ask questions interactively; type `quit` to exit.

## Possible next steps

- Migrate from the deprecated `google.generativeai` package to the current `google.genai` library.
- Replace `eval` with a proper, safe math-expression parser.
- Add richer tools — including wiring in a RAG system as one of the agent's tools, so "ask my documents" becomes an action the agent can choose.
- Auto-retry with backoff on API rate limits.
