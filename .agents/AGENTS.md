# Workspace Rules

## Session Context Initialization
Before starting ANY work in a session, you MUST read the `CONTEXT.md` file located at the repository root. After reading it, summarize the current state of the project back to the user before proceeding with any planning or execution. This ensures work doesn't duplicate or contradict prior decisions. Do this proactively without the user prompting you to.

## Session Context Handoff
At the end of every session (even if the session ends abruptly or the task wasn't fully finished), you MUST update the `CONTEXT.md` file. It is an append-only log. Add a new entry with:
- Date/timestamp
- What was done this session (files changed, experiments run, results obtained)
- Current state of the core claim
- Any decisions made and why
- Any open questions, blockers, or things flagged for human review
- Exact next steps for whoever picks this up next
