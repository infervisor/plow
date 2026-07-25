# plow

## Plans & research

- Implementation plans, design proposals, and research notes live in `plans/`.
- This directory is **local-only** (gitignored) — it is a scratchpad for coding
  agents and is not checked into version control.
- Read the relevant file in `plans/` before starting work, and update it as
  decisions change.

# CLAUDE.md

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop. Name what's confusing.
- Always provide a high level design plan

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No “flexibility” that wasn't requested.
- If 200 lines could be 50, rewrite it.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice dead code, mention it — don't delete it.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- “Fix the bug” → “Reproduce it in a test, then fix”
- “Refactor X” → “Ensure tests pass before and after”


Rules:
- Use very short sentences
- Remove filler words (the, a, an, is, are, etc. where possible)
- No politeness (no "sure", "happy to help")
- No long explanations unless asked
- Keep only meaningful words
- Prefer symbols (→, =, vs)
- Output dense, compact answers


## Development

- check for relevant rust crates
- for launching terminal tasks Use "nix develop"
- use rust types 
- focus on code reuse and consistency

## runtime and kernels

- for plowrt performance is more important
- every microsecond is important