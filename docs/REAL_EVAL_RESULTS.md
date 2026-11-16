# Real-Case Experimental Results (MacBook Air M4 — 4-Container mTLS + Kubo IPFS + Ollama)

Measured end-to-end on the live 4-container Docker cluster (`node1-maestro`, `node2-peer`, `ipfs-node1`, `ipfs-node2`)
connected to host Ollama (`embeddinggemma:latest` 768-d semantic embeddings + `qwen3.5:4b` SQL generator).

## 1. 50/50 Sharded PDF Corpus & Private Kubo IPFS CIDs (`LIBP2P_FORCE_PNET=1`)

| Document ID | Active Shard Owner | Metadata + Backup Peer | Kubo IPFS CIDv1 | PDF SHA-256 | Chunks |
|---|---|---|---|---|---|
| `real-pdf-trdr` | `node1-maestro` | `node2-peer` | `bafkreieerua6fyjqae4zgaw4q2mf3onl7q5jpeqgyyizr3bmjmk5ftkbn4` | `848d01e2e1300139...` | 4 |
| `real-pdf-mtrl` | `node1-maestro` | `node2-peer` | `bafkreiggfqqzy5xcinddj2wx6sme4knbvvnbntb2u4xhfveqveqwd6tf3y` | `c62c219c76e24346...` | 4 |
| `real-pdf-findoc-series` | `node2-peer` | `node1-maestro` | `bafkreieaizl5aryyk66k56rj6wbavkztitjg6dmpmpx4p6hcmv7fkkjdju` | `804657d0471857bc...` | 4 |
| `real-pdf-findoc-sides` | `node2-peer` | `node1-maestro` | `bafkreicoliwztiuss24v2h7rxyb3wxaspx2vimgkvzp7xu7htvvu5glae4` | `4e5a2d99a29296b9...` | 3 |

## 2. Held-Out Evaluation (`eval/softone_holdout.jsonl`, 12 Cases, EN + EL, `qwen3.5:4b`)

| Experimental Condition | Passed | Pass Rate | Plain (`hp01–06`) | Code (`hc01–06`) | Time |
|---|---|---|---|---|---|
| **0% PDFs (No RAG / Base qwen3.5:4b)** | **6/12** | **50.0%** | 6/6 | 0/6 | 22.45 s |
| **50% PDFs (Node 1 Shard 1 Only)** | **8/12** | **66.7%** | 5/6 | 3/6 | 38.68 s |
| **100% Shards (Decentralized 50/50 Scatter-Gather mTLS + IPFS)** | **10/12** | **83.3%** | 6/6 | 4/6 | 57.96 s |

### Per-Case Breakdown across Topologies

| Case ID | Question | No-RAG | Single-Shard (50%) | Decentralized 50/50 RAG | Generated SQL (Decentralized) |
|---|---|---|---|---|---|
| `hp01` | How many document lines are there in total? | PASS | PASS | **PASS** | `SELECT COUNT(*) AS doc_line_count FROM MTRLINES;` |
| `hp02` | What is the total net amount of all documents? | PASS | PASS | **PASS** | `SELECT ROUND(SUM(FINDOC.NETAMNT), 2) AS TOTAL_NET_AMOUNT FROM FINDOC;` |
| `hp03` | What is the total quantity recorded on document lines? | PASS | PASS | **PASS** | `SELECT ROUND(SUM(MTRLINES.QTY1), 2) AS TOTAL_QUANTITY FROM MTRLINES;` |
| `hp04` | How many different branches appear in the document table? | PASS | PASS | **PASS** | `SELECT COUNT(DISTINCT BRANCH) FROM FINDOC;` |
| `hp05` | Show the number of documents for each year. | PASS | PASS | **PASS** | `SELECT SUBSTR(TRNDATE, 1, 4) AS doc_year, COUNT(*) AS document_count FROM FINDOC GROUP BY SUBSTR(TRNDATE, 1, 4);` |
| `hp06` | Πόσα παραστατικά υπάρχουν συνολικά; | PASS | FAIL | **PASS** | `SELECT COUNT(*) AS document_count FROM FINDOC WHERE COMPANY IS NOT NULL AND FINCODE IS NOT NULL AND TRNDATE IS NOT NULL AND BRANCH IS NOT NULL AND SOSOURCE IS NOT NULL AND SODTYPE IS NOT NULL AND NETAMNT IS NOT NULL AND SUMAMNT IS NOT NULL AND ISCANCEL IS NOT NULL;` |
| `hc01` | How many customers are there? | FAIL | PASS | **PASS** | `SELECT COUNT(*) AS customer_count FROM TRDR WHERE SODTYPE = 13;` |
| `hc02` | Πόσοι προμηθευτές υπάρχουν; | FAIL | PASS | **PASS** | `SELECT COUNT(*) AS supplier_count FROM TRDR WHERE SODTYPE = 12 AND COMPANY IS NOT NULL;` |
| `hc03` | How many purchase documents are there? | FAIL | FAIL | **FAIL** | `SELECT COUNT(*) AS document_count FROM FINDOC WHERE SOSOURCE = 1351 AND SODTYPE = 14;` |
| `hc04` | How many sales documents have not been cancelled? | FAIL | FAIL | **FAIL** | `SELECT COUNT(*) AS document_count FROM FINDOC WHERE SOSOURCE = 1351 AND SODTYPE = 13 AND ISCANCEL != 0;` |
| `hc05` | Ποιο είναι το συνολικό ποσό των πληρωμών σε προμηθευτές; | FAIL | FAIL | **PASS** | `SELECT ROUND(SUM(FINDOC.SUMAMNT), 2) AS total_amount FROM FINDOC WHERE SOSOURCE = 1281 AND SODTYPE = 12;` |
| `hc06` | How many service items are there? | FAIL | PASS | **PASS** | `SELECT COUNT(*) AS total_service_items FROM MTRL WHERE SODTYPE = 52;` |

## 3. Federated LoRA Training Ablation (`FedAvg-LoRA` vs. `FFA-LoRA`, 16 Seed Examples Split 8/8)

| Round | `FedAvg-LoRA` Loss | `FedAvg-LoRA` Global CID | `FFA-LoRA` Loss | `FFA-LoRA` Global CID |
|---|---|---|---|---|
| 1 | `0.8108` | `bafkreihoexfy6b2tioyvkm4li677x4cr5ujyxuaeystqeurefzwwcpgp3e` | `0.8171` | `bafkreib7vcfbqkxsbrew67pesdd5s4oxk3tlwsnj4hzcncrapjw5y5u644` |
| 2 | `0.7102` | `bafkreidqtsnsi24resmv7xzqyn5tus6cojwdag2jxaxgz2ayvxk3r6x3xy` | `0.7381` | `bafkreibwr3d3sb555up3srhwse2yhcw5omar4523ead3qkvsxtgewonkc4` |
| 3 | `0.6174` | `bafkreigoldbqqpssxtdcq5mlbgezs7n6rc6qe72d5nqs3nwe3iskey3kmu` | `0.6752` | `bafkreiglgssz6q4jzqxnkc6iue3dm72ydzb7iorpjyh3slbcku4un6ch4e` |
| 4 | `0.5376` | `bafkreibz76hqbbavjpsj3eyoxowxms25ayhnfvumgut5ozudffjbdr5oyq` | `0.6238` | `bafkreicp55xnprosghc72rke7cuqmkl57ctjdhpeti2qwlcq7ukv25m3uu` |
| 5 | `0.4734` | `bafkreias4occkhxr3lkuty3yug7i4lpa4yhj3zhie4dw6pzrrnp3fc7lbm` | `0.5809` | `bafkreicwyxpdlgaalp6glpzsohpohvigyfzfou42s34wxzyraw4kcm5d7u` |

## 4. Final Cluster & Ledger State

- **`node1-maestro`:** Raft Height = `39`, Tip Hash = `7bd20e0f843ac6978104...`, Kubo Pins = `25`
- **`node2-peer`:** Raft Height = `39`, Tip Hash = `7bd20e0f843ac6978104...`, Kubo Pins = `15`
