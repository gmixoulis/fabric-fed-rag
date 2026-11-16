---
title: "Prompt examples and test questions must be different strings, and old scores are inflated"
author: Gmichoulis
date: 2026-11-16
branch: main
cost: { minutes: 148, tokens: 906827 }
id: l-c806cd90
scope: repo
status: active
src: agent
touches:
  - ".python-version"
  - AGENTS.md
  - README.md
  - TUTORIAL.md
  - deploy/README.md
  - deploy/macos/local.llama.gpulimit.plist
  - deploy/macos/local.llama.rpc.plist
  - deploy/macos/start-rpc.sh
  - deploy/systemd/llama-split.service
  - deploy/systemd/softone-chat-public.service
---

## Decided
Whenever examples are added to the SQL prompt or cases are added to the test sets, check the two for overlap. Scores from before the examples were replaced are not reported as model quality.

## Tried and ruled out
Asking one coding agent, in one brief, to add both off-topic prompt examples and off-topic test cases: it reused the same sentences. Five of 25 real test questions appeared verbatim in the prompt, so a perfect refusal score was partly copying.

## Watch out
- The test set also had wrong ground truth: a question asking for a total was scored against two separate numbers, and "items" was scored against every MTRL row instead of goods and services.
- The language scorer counted a Greek supplier name inside an English answer as Greek; it now removes data values first.
- Ground truth is computed from the real data by the reviewer. The coding agent has no data access and its expected values are unchecked.
- When a case fails, decide first whether the system, the test or the infrastructure is at fault.

In the code: src/softone_rag/graph.py build_prompt (examples), eval/real_cases.jsonl, eval/softone_holdout.jsonl, src/softone_rag/real_eval.py.
