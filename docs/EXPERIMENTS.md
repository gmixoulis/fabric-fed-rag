# Experiments and measurements

Every number here was measured on the hardware below on the date given, unless
it is marked *not measured* or *from notes*. Nothing is extrapolated. Read the
"Threats to validity" section before quoting any figure.

## Hardware

| Node | Role | CPU / RAM | GPU | Backend |
|---|---|---|---|---|
| PC (Linux 7.0, home server) | front end, app, database, embeddings | 16 threads, 15 GiB RAM, 11 GiB swap | GTX 1660 SUPER, 6 GB (about 5.2 GB free) | llama.cpp b12426, Vulkan |
| MacBook Air M4 | RPC worker | 10 cores (4P + 6E), 16 GB unified | Apple GPU, Metal; limit raised from about 12.1 GB to 14 GB (`iogpu.wired_limit_mb=14336`) | llama.cpp b12426, Metal |

Link: same LAN, Tailscale direct connection, about 2 ms round trip. The two
llama.cpp builds are the identical release (RPC requires matching versions).

## E1. Inference speed: split versus single machine (2026-11-16)

Tool: `llama-bench`, default prompt and generation lengths, one run each unless noted.

| Setup | Model file | Generation tok/s | Prompt tok/s | Notes |
|---|---|---|---|---|
| PC + Mac split | gemma4 26B UD-Q3_K_M, 11.8 GB | 13.3 | 248 | Mac GPU limit 12.1 GB; PC 4.7 GB + Mac 8.5 GB |
| PC + Mac split | gemma4 26B UD-Q4_K_M, 15.8 GB | about 17.5 | about 271 | Mac GPU limit 14 GB; PC 4.9 GB + Mac 12.4 GB. **Production setup.** |
| Mac alone | gemma4 26B UD-Q3_K_M | 12.4 | 170 | Q4 does not fit on the Mac alone |
| PC alone | gemma4 26B (any) | does not fit | does not fit | 6 GB card |
| PC + Mac split | gpt-oss-20b MXFP4, 11.3 GB | about 20 | 165 | rejected on quality, see E4 |
| PC alone, Ollama | qwen3.5:4b | about 65 | not recorded | from notes dated 2026-10-09, not re-measured |

Other observations:

- PC thread count 8 versus 16: generation unchanged within noise, prompt
  processing about 15% slower with 16. Kept 8.
- Cold start (weights sent to the Mac for the first time): about 2 min 45 s.
  Later starts with the Mac's on-disk weight cache: 103 s.
- Directly after a macOS update, with the Mac busy (load average 88),
  generation dropped to about 13 tok/s on the Q4 split.
- The Q4 split leaves the Mac at 4–5% free memory with swap in use (macOS
  memory pressure "warning").

Why the split beats the Mac alone: the PC's GPU has roughly three times the
memory bandwidth, so the layers it holds run faster than they would on the M4,
and this outweighs the per-token network round trip on a 2 ms link.

## E2. Embedding models for Greek/English recall (2026-11-16)

Task: 16 questions (cross-language and paraphrase) against 28 stored texts
(taught facts and saved question/SQL examples). Metric: rank of the correct
text by cosine similarity.

| Model (Ollama) | Top-1 | Top-3 | MRR | Time for 28 texts |
|---|---|---|---|---|
| **bge-m3** | 15/16 | 16/16 | 0.969 | 2.3 s |
| embeddinggemma | 14/16 | 15/16 | 0.922 | 1.0 s |
| qwen3-embedding:4b | 14/16 | 15/16 | 0.915 | 18.2 s |
| qwen3-embedding:0.6b | 10/16 | 14/16 | 0.762 | 12.0 s |

Decision: bge-m3. The margin over embeddinggemma is one question out of 16.

## E3. Federated LoRA training, plumbing test (2026-11-16)

Base: Qwen3.5-4B (PC: 4-bit bitsandbytes on CUDA inside `train-safe`; Mac:
bfloat16 on MPS, memory fraction 0.75, RSS watchdog at 13 GB).
LoRA r=4, alpha=8, dropout 0.05, targets `q_proj`/`v_proj`, lr 1e-4,
8 local steps per round, batch 1, FedAvg weighted by examples.
Data: 16 schema-only question/SQL pairs, split 8/8.

| Round | PC loss first → last | PC time | Mac loss first → last | Mac time | Round time |
|---|---|---|---|---|---|
| 1 | 0.93 → 0.56 | 196 s | 0.55 → 0.37 | 149 s | 205 s |
| 2 | 1.04 → 0.46 | 196 s | 0.38 → 0.75 | 147 s | 207 s |
| 3 | 0.38 → 0.98 | 206 s | 0.90 → 0.35 | 146 s | 217 s |

Verified: all 32 averaged LoRA tensors equal the mean of the two client
adapters; each round starts from the previous global adapter; PC training
scope peaked at about 1.3 GiB against a 6 GiB cap; the Mac stayed at 72% free.

**This run shows the pipeline works. It does not show learning.** With batch
size 1 and 8 examples per client, the first and last loss of a round come from
different examples. No held-out evaluation was run on this adapter.

## E4. Model choice on the production task (2026-11-16)

Three hand-checked SoftOne questions through the full pipeline:

| Model | SQL | Combined (multi-step) answer | Greek question |
|---|---|---|---|
| gemma4 26B Q4 split | correct | good | correct after the recall fix (E6) |
| gpt-oss-20b split | correct | failed; fell back to a bare list | failed |

Three questions is a smoke test, not an evaluation.

## E5. Real-pipeline evaluation (2026-11-16)

`softone-rag eval --real --database softone_reputation`: 25 cases through the
production path (gemma4 26B Q4 split, bge-m3, taught facts, multi-step
analysis, read-only snapshot). Memory is copied for the run; the real memory
file is never written.

Case kinds: 12 `sql` (result rows must equal the gold query's rows), 7 `refuse`
(off-topic or hostile; must refuse without running SQL), 2 `empty_ok` (a name
that is not in the data), 4 `analysis` (multi-step; every expected number must
appear in the answer). 8 cases are in Greek.

| Run | Score | What changed before it | Valid? |
|---|---|---|---|
| 1 | 16/25 | first run | No. Model server killed by earlyoom after case 16; 9 cases never ran |
| 2 | 19/25 | rerun | No. Killed again; 6 cases never ran |
| 3 | 19/25 | `--load-mode none` (no mmap) | No. Killed again at the same case |
| 4 | 20/25 | `--cache-ram 1024` | First complete run. sql 12/12, refuse 7/7, empty_ok 0/2, analysis 1/4 |
| 5 | 23/25 | name-lookup rule, language-check fix, two gold corrections, prompt examples no longer copy test questions | Yes. sql 12/12, refuse 7/7, empty_ok 2/2, analysis 2/4. Server peak RAM 4.5 GB, median 33 s per case |
| 6 | 27/29 | fixes for over-refusal of invoice questions, SoftOne read as a trader name, untyped "latest invoice", Greeklish replies; 4 new cases (2 Greeklish); automatic prompt/test leakage test | Yes. sql 16/16, refuse 7/7, empty_ok 2/2, analysis 2/4 (same two ambiguous questions as run 5). Median 34 s per case |

Run 4 failure analysis: two gold errors (a question asking for a total was
scored against two separate numbers; "items" was scored against all 1,523 MTRL
rows instead of the 21 goods and services), one scorer bug (an English answer
quoting a Greek supplier name was flagged as Greek), and two genuine model
failures (exact-match name lookup with `COUNT(*)`, answered "0").

Run 5 failure analysis: both failures are ambiguous questions, and the model
read them differently from run 4. "How many customers and suppliers are there
in total?" was answered 58 in run 4 and "44 suppliers and 14 customers" in
run 5; the gold accepts only one reading. "Πόσοι πελάτες και πόσα είδη" counted
goods and services (21) in run 4 and goods only (18) in run 5. So answers to
underspecified questions are not stable between runs, and whether "είδη"
includes services is a business definition nobody has taught. The questions
were left as they are: rewording tests until they pass would hide this.

**Leakage found and removed.** Five of the 25 test questions appeared verbatim
as examples in the SQL prompt (r01, r02, r04, r06, e01), so run 4's 7/7 on
refusals is inflated. The examples were replaced with different ones before
run 5. Any figure from runs 1–4 must not be reported as model quality.

## E6. Defects found by measurement

| Symptom | Root cause | Fix |
|---|---|---|
| Greek question refused although the codes were taught | Saved examples filled every recall slot and pushed the taught facts out of the prompt | Facts pinned to the retrieved tables are always included |
| Model server killed mid-evaluation (three times) | earlyoom sends SIGTERM to the largest process when RAM < 10% and swap < 10%. Host swap was full; llama-server showed 7 GB RSS | see next two rows |
| llama-server RSS about 7 GB with all layers on GPUs | The 15.8 GB GGUF stayed memory-mapped | `--load-mode none`: RSS 1.7 GB after load |
| RSS still grew to 7.3 GB during the run | Prompt cache defaults to 8192 MiB of host RAM | `--cache-ram 1024`: peak 4.5 GB under a 6 GB cgroup cap |
| Chat down about 30 min | `--no-mmap` is not a valid flag in build b12426; the unit restart-looped | Correct flag is `--load-mode none`. Check `--help` of the pinned build before editing a unit |
| Off-topic refusal leaked an internal token | Refusal text rendered the raw reason `CANNOT_ANSWER` | Scope message in the question's language with three example questions |
| "The query returned no rows." | No explanation of what was searched | Deterministic explanation built from the SQL |

Run 6 note: the owner found the over-refusal by asking the live chat for "the
last invoice ever built from softone", which the 25-case set did not cover. A
refusal-only test set rewards a model for refusing; in-scope questions phrased
loosely (and in Greeklish) are now part of the set.

## E7. Invoice drafts against AADE's myDATA test environment (2026-11-16)

`softone-rag invoice-draft --database softone_reputation --findoc N [--send-test]`
rebuilds an existing SoftOne sales document as a myDATA v2.0.2 InvoicesDoc,
validates it against the official XSD, reconciles it with the database and,
in test mode, submits it to `mydataapidev.aade.gr`.

| Step | Result |
|---|---|
| Rebuild + reconcile, documents 458, 457, 321 | net, VAT, gross and line counts equal the database for all three |
| Document 459 (0% VAT) | refused: no VAT exemption category in the data, and none is guessed |
| First test submission | rejected: 105 (test user may issue only for its own ΑΦΜ), 204 (currency mandatory for type 2.1), 205 (quantity forbidden for 2.1), 247 (itemCode forbidden for 2.1) |
| After fixing per-type field rules and substituting the test user's ΑΦΜ as issuer in test mode | **accepted**: statusCode Success, MARK and UID returned; PDF carries MARK, UID and the QR from `qrUrl` |

Defects only visible against real data and the real API: tax office and VAT
tables have no company column; the customer's tax office is stored as text;
currency and country are internal ids (ISO codes in `INTERCODE`); a sqlcmd
flag conflict; and the invoice-type field rules above. The client's company
address is empty in this backup, so the printed Art. 9 identity check fails
for the issuer; the PDF says so.

## Threats to validity

- **Single runs.** Speeds are one `llama-bench` run each; no variance is reported.
- **Quantization is confounded with topology.** The split runs Q4 and the
  Mac-alone baseline runs Q3. Speed and accuracy differences between them
  cannot be attributed to the split alone. Accuracy of Mac-alone Q3 was not measured.
- **Tiny evaluation set.** 25 cases, one database, written by the same people
  who built the system, with gold SQL drafted by a model and checked by hand.
- **Leakage history.** See E5. Only runs after the prompt examples were
  changed are free of verbatim test questions.
- **One client.** Three backups of one SoftOne installation; codes were
  verified against that installation's own series names.
- **Federated data is not naturally partitioned.** Both shards come from one
  seed file split at random; there is no real data-locality constraint.
- **Network.** One LAN, 2 ms. No test over Wi-Fi-only, WAN or a slower link.
- **The old 87.5% figure** in `eval/report.md` is from scripted model
  responses on synthetic data. It measures the pipeline's controls, not a model.

## Experiments still needed for a paper

1. Same 25+ cases on: Mac-alone Q3, split Q3, split Q4 (separates topology from quantization).
2. Repeat counts (at least 3) and confidence intervals for speed and pass rate.
3. Ablations: taught facts off, saved examples off, multi-step analysis off, guard-only refusals.
4. FedAvg versus centralized LoRA on the same examples, scored on the held-out set, once at least 200 confirmed examples exist.
5. Small model (qwen3.5:4b) with and without the adapter on the held-out set.
6. Link sensitivity: LAN versus Wi-Fi versus injected latency.
7. A larger, independently written test set, with Greek/English balance.
