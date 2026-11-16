# Research log

How this system came to look the way it does: the question at each step, what
was tried, what was measured, what was rejected and why. Numbers are in
`EXPERIMENTS.md`; structure is in `ARCHITECTURE.md`. Dates are 2026.

## 0. Starting point (before 13 Nov)

`softone-rag` was a learning project: retrieval plus model-generated SQL over a
synthetic database, with a simulated evaluation. Three real SoftOne SQL Server
backups had been restored in Docker and exported to read-only SQLite snapshots
(four tables, selected columns). A small local model (qwen3.5:4b on a 6 GB
GTX 1660 SUPER) answered simple questions. Type codes for customers, sales and
cancellations were unverified, so the guard blocked every question that
depended on them.

A first attempt at QLoRA fine-tuning on the PC alone (three experiment
directories) left pre-training scores and no adapter. The two cases it failed
were "undefined sales" and "undefined customers": a knowledge gap, not a
capacity gap.

## 1. "Federated" meant two different things (13–15 Nov)

The owner asked to use the MacBook Air M4 together with the PC, "distributed
and federated". Two readings:

- **Federated learning**: train one model on several machines that each hold
  private data, sharing only weight updates.
- **Distributed inference**: run one model that fits on neither machine by
  placing its layers on both.

The data sits in one place, so federated learning has no privacy motive here.
What the hardware lacked was memory for a better model. That pointed to
distributed inference first, with federated training as a later experiment.

**Ollama cannot do this.** Ollama issue #4643 (open since May 2024) asks for
llama.cpp's RPC backend. A comment claiming `ollama rpc` and
`OLLAMA_RPC_SERVERS` exist was contradicted in the same thread (the command is
absent in 0.30.0-rc15). The `shadoq/ollama-cluster` fork was tested only on two
identical Linux/CUDA machines with Docker and reports the split as *slower*
when a model fits on one node (gpt-oss:120b 42.2 → 34.1 tok/s).

Decision: use llama.cpp directly. `llama-server` on the PC, `ggml-rpc-server`
on the Mac, identical release on both (RPC needs matching builds), official
prebuilt binaries (Vulkan on the PC because no CUDA toolkit is installed, Metal
on the Mac). The RPC server has no authentication, so it binds to the Mac's
Tailscale address only.

Practical obstacles, each cost time and none is in the code:
two Tailscale node entries for what looked like one Mac; Remote Login not
actually enabled; a hand-typed `authorized_keys` line with a letter O where the
key had a zero (found by comparing key fingerprints); port 8080 already in use;
the RPC binary renamed to `ggml-rpc-server`.

## 2. Which model, and where (16 Nov)

Memory budget: about 5.2 GB on the PC GPU plus about 12.1 GB on the Mac
(macOS caps GPU memory at roughly 75% of RAM).

- qwen2.5:32b (about 20 GB at Q4) and gemma4 26B at Q4 (about 16 GB) did not fit.
- gemma4 26B at Q3 (11.8 GB) fit: 13.3 tok/s.
- Raising the Mac's GPU limit to 14 GB (`iogpu.wired_limit_mb=14336`, applied
  at boot by a LaunchDaemon) made Q4 fit: about 17.5 tok/s, at the cost of the
  Mac running at 4–5% free memory.

Counter-intuitive result: the larger Q4 split is *faster* than the smaller Q3
on the Mac alone (12.4 tok/s). The PC's GPU has about three times the memory
bandwidth, so every layer moved onto it is a layer that runs faster. On a 2 ms
link this outweighs the per-token round trip. gemma4 26B is a
mixture-of-experts model that reads about 4B parameters per token, which is why
a 16 GB model generates at a usable speed at all.

Rejected:
- **gpt-oss-20b**: slightly faster to generate (about 20 tok/s), slower on
  prompts, and worse on the task (failed the combined answer and a Greek question).
- **Using all CPU threads**: no gain; GPUs do the work.
- **Mac alone**: slower and limited to Q3.
- **A larger cloud model** (Inkling 2 via Ollama Cloud): breaks the local-only rule.

Still honest: per-answer speed is four to five times *slower* than the 4B model
on the PC alone (about 65 tok/s from October notes). The split buys model
quality, not speed.

## 3. The model was not the bottleneck; knowledge was

The guard refused "how many customers" because nobody had established what
`TRDR.SODTYPE = 13` means. Fine-tuning cannot fix that honestly: the meaning of
a code is a fact about one installation.

Decision: a **memory of human-taught facts**. Each fact carries a table, column
and literal value. The guard allows a comparison on a type-code column only for
literals a human has taught. Fifteen facts were taught after verifying each
against SoftOne's own series and trader names in the restored databases (never
guessed). One code (1361) was left untaught because its meaning was unclear at
the time; the series names later showed it is the "transmitted" invoice series.

Caveat recorded in the fact itself: sales code 1351 also covers returns, and
the snapshots lack the field that separates them, so sales totals are gross.

Saved question/SQL pairs are a second kind of memory: unconfirmed hints until a
person presses 👍.

## 4. Multi-step answers, language, refusals

- **Analysis**: a hard question is split into at most three sub-questions; each
  runs through the same guarded path; the combined answer is rejected if it
  contains a number absent from the query results. The check is plain code.
- **Language**: the reply follows the question's language (Greek or English),
  detected from the share of Greek letters. No model call.
- **Off-topic**: the system answers only from the data. A question about a
  brand on its own is refused; "is that brand a customer?" is answered with SQL.
  Refusals explain the scope and offer example questions; they never show
  internal tokens.

## 5. Measuring the real thing

The existing 87.5% came from scripted responses on synthetic data. A real
evaluation was built: the production `ask` function, the real model, a real
snapshot, a private copy of memory. Twenty-five cases of four kinds.

What the first runs taught (details in `EXPERIMENTS.md` E5, E6):

1. **Infrastructure failed before the model did.** Three runs died at the same
   point. earlyoom killed the model server because the host's swap was full and
   llama-server's resident memory grew to 7 GB: first from the memory-mapped
   model file, then from an 8 GiB default prompt cache. Two flags fixed it.
2. **A wrong flag took the chat down for half an hour.** The fix above was first
   written as `--no-mmap`, which the pinned build does not accept.
3. **Three of five failures were the test's fault**: two gold answers encoded a
   wrong reading of the question, and the language scorer counted a Greek
   supplier name inside an English sentence.
4. **The test had leaked into the prompt.** Five test questions were used
   verbatim as prompt examples by the same change that added them to the test
   set. The examples were replaced. Scores before that are not model quality.

Lesson kept: a brief that asks one agent for both "prompt examples" and "test
cases" will reuse the same strings. Write them separately and check for overlap.

## 6. Embeddings and the vector store

bge-m3 won a 16-question Greek/English recall test by one question
(`EXPERIMENTS.md` E2). Vectors live in the same SQLite file as the text;
similarity is a brute-force cosine over a few dozen rows.

Rejected:
- **A vector database** (Qdrant and similar): no benefit below roughly 10^5 rows.
- **A cloud embedding model**: every question and fact would leave the network.

A retrieval bug mattered more than the choice of model: saved examples filled
every recall slot and pushed taught facts out of the prompt, so a Greek question
was refused although its codes were known. Facts tied to the retrieved tables
are now always included.

## 7. Federated training, kept in proportion

Built and run end to end: FedAvg over LoRA adapters, PC on CUDA inside a
memory-limited cgroup (`train-safe`), Mac on MPS with a memory-fraction cap and
an RSS watchdog. Three rounds on 16 examples completed; averaging was verified
tensor by tensor.

What it is not: evidence of learning (16 examples, batch 1), or faster than
training on the Mac alone. Its honest motivations here are using both machines
and, later, several businesses whose data must stay on their own machines.

Supporting pieces: export of 👍-confirmed answers as training data (re-checked
by the guard, held-out questions excluded), a held-out set with before/after
scoring, and a nightly cycle that trains only with at least 20 new confirmed
examples and keeps an adapter only if it scores strictly higher.

The mixed-backend detail that cost the most time: the coordinator needs the app
environment (to build prompts) and averaging needs the training environment
(torch). Neither environment has the other's packages.

## 8. Safety decisions

- The model never holds a database connection. Three independent layers stop a
  write: text-only model, single-`SELECT` guard, read-only connection.
- Public access goes to a separate port with a login; the loopback port has no
  login and is never a proxy target. Deciding by port, not by the `Host` header,
  because a local reverse proxy also connects from 127.0.0.1.
- MCP was considered as a safety boundary. It is a transport, not a boundary:
  safety comes from exposing only read-only, guarded tools. Worth adding when
  n8n or another agent needs access.
- Evaluation runs on a copy of memory so tests cannot teach the system.

## 9. Rejected or deferred, with reasons

| Idea | Outcome | Reason |
|---|---|---|
| Ollama-native distributed inference | rejected | not implemented upstream |
| `ollama-cluster` fork | rejected | untested on CUDA + Metal; maintenance risk |
| jOpenAgent (Java agent framework) | rejected for this project | Python codebase with equivalent features; its code-execution mode is riskier than guarded SQL |
| gpt-oss-20b | rejected | worse answers on the task |
| Vector database | rejected | scale does not need it |
| Cloud embeddings / cloud models | deferred | local-only rule; a per-step opt-in later |
| Fine-tuning to "learn" codes | rejected | codes are facts, taught and verified |
| AI issuing tax documents | rejected | legal: from 1 Feb 2027 only a certified provider or AADE's own apps may issue B2B invoices; the AI drafts, SoftOne issues |

## 10. Open questions (candidate paper contributions)

1. Heterogeneous split inference (CUDA-class + Metal over a LAN) can beat the
   larger-memory node alone. Needs the controlled comparison in `EXPERIMENTS.md`.
2. Verified, human-taught facts coupled to a syntactic guard as an alternative
   to fine-tuning for installation-specific ERP semantics. Needs ablations.
3. Number-grounded multi-step answers: a deterministic check that every figure
   in the prose appears in query results.
4. Operating cost of local LLM serving on consumer hardware: the failures that
   mattered were memory accounting and process supervision, not model quality.
5. Evaluation hygiene in agent-built systems: prompt/test leakage introduced by
   the same automated change that created the tests.
6. Federated adapters across unlike backends (4-bit CUDA and bfloat16 MPS):
   does averaging adapters trained against differently quantized bases hurt?

## 11. How the work was done

One orchestrating model wrote briefs and reviewed every change; a second,
cheaper coding agent wrote the code; research agents read external sources.
Every delegated change was re-tested by the orchestrator before acceptance.
Ground truth for tests was always computed from the real data by the
orchestrator, never trusted from the coding agent, which had no data access.
Review found, among others: test cases that returned zero rows on the chosen
snapshot, an averaging step in the wrong environment, and the leakage above.
