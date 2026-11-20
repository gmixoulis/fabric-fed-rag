#!/usr/bin/env python3
"""Export all FabricFedRAG experimental results, LoRA ablations, IPFS CIDs, and paper comparisons to Excel (.xlsx) & CSV."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

_ROOT = Path(__file__).resolve().parents[1]


def _col_letter(col_idx: int) -> str:
    """Convert 0-based column index to Excel column letter (A, B, ..., Z, AA...)."""
    result = ""
    n = col_idx + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def _sheet_xml(rows: list[list[object]]) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <sheetData>",
    ]
    for r_idx, row in enumerate(rows, 1):
        lines.append(f'    <row r="{r_idx}">')
        for c_idx, val in enumerate(row):
            cell_ref = f"{_col_letter(c_idx)}{r_idx}"
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                lines.append(f'      <c r="{cell_ref}"><v>{val}</v></c>')
            else:
                text = escape(str(val if val is not None else ""))
                lines.append(
                    f'      <c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        lines.append("    </row>")
    lines.append("  </sheetData>")
    lines.append("</worksheet>")
    return "\n".join(lines).encode("utf-8")


def write_multi_sheet_xlsx(out_path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    """Write a standards-compliant OpenXML .xlsx workbook with multiple named worksheets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '  <Default Extension="xml" ContentType="application/xml"/>',
        '  <Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>\n'
        "</Relationships>"
    )

    wb_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "  <sheets>",
    ]
    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (name, _rows) in enumerate(sheets, 1):
        safe_name = escape(name[:31])
        wb_lines.append(f'    <sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>')
        wb_rels.append(
            f'  <Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    wb_lines.append("  </sheets>")
    wb_lines.append("</workbook>")
    wb_rels.append("</Relationships>")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", "\n".join(wb_lines))
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))
        for idx, (_name, rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def export_all_from_json(raw_json_path: Path, results_dir: Path) -> Path:
    data = json.loads(raw_json_path.read_text())

    # 1. Summary sheet
    summary_rows: list[list[object]] = [
        [
            "Experimental Condition",
            "Passed Cases",
            "Total Cases",
            "Pass Rate (%)",
            "Plain SQL Passed (out of 6)",
            "Code-Guarded Passed (out of 6)",
            "Total Latency (s)",
        ]
    ]
    for mode_key in ("no_rag", "single_shard_node1", "decentralized_50_50_rag"):
        info = data["eval_summary"][mode_key]
        hp_pass = sum(1 for c in info["cases"] if c["id"].startswith("hp") and c["passed"])
        hc_pass = sum(1 for c in info["cases"] if c["id"].startswith("hc") and c["passed"])
        summary_rows.append(
            [
                info["label"],
                info["passed"],
                info["total"],
                info["pass_rate_pct"],
                hp_pass,
                hc_pass,
                info["seconds"],
            ]
        )

    # 2. Per-case holdout sheet
    holdout_rows: list[list[object]] = [
        [
            "Case ID",
            "Language",
            "Kind",
            "Question",
            "Gold SQL",
            "No-RAG Result",
            "Single-Shard (Node 1) Result",
            "Decentralized All-Shard RAG Result",
            "Decentralized Generated SQL",
            "Decentralized Latency (s)",
        ]
    ]
    no_cases = data["eval_summary"]["no_rag"]["cases"]
    s1_cases = data["eval_summary"]["single_shard_node1"]["cases"]
    dec_cases = data["eval_summary"]["decentralized_50_50_rag"]["cases"]
    for idx, dec in enumerate(dec_cases):
        lang = "EL" if any("\u0370" <= ch <= "\u03ff" for ch in dec["question"]) else "EN"
        kind = "Plain Schema" if dec["id"].startswith("hp") else "Code-Guarded (RAG Required)"
        holdout_rows.append(
            [
                dec["id"],
                lang,
                kind,
                dec["question"],
                dec["gold_sql"],
                "PASS" if no_cases[idx]["passed"] else "FAIL",
                "PASS" if s1_cases[idx]["passed"] else "FAIL",
                "PASS" if dec["passed"] else "FAIL",
                dec["generated_sql"],
                dec["elapsed_s"],
            ]
        )

    # 3. Federated LoRA ablation sheet
    lora_rows: list[list[object]] = [
        [
            "Round",
            "FedAvg Shard1 First Loss",
            "FedAvg Shard1 Last Loss",
            "FedAvg Shard2 First Loss",
            "FedAvg Shard2 Last Loss",
            "FedAvg Global Weighted Loss",
            "FedAvg Global Kubo IPFS CID",
            "FFA-LoRA Shard1 First Loss",
            "FFA-LoRA Shard1 Last Loss",
            "FFA-LoRA Shard2 First Loss",
            "FFA-LoRA Shard2 Last Loss",
            "FFA-LoRA Global Weighted Loss",
            "FFA-LoRA Global Kubo IPFS CID",
        ]
    ]
    for r_idx in range(len(data["lora_ablation"]["fedavg"])):
        fa = data["lora_ablation"]["fedavg"][r_idx]
        ff = data["lora_ablation"]["ffa"][r_idx]
        fa_c1, fa_c2 = fa["clients"][0], fa["clients"][1]
        ff_c1, ff_c2 = ff["clients"][0], ff["clients"][1]
        lora_rows.append(
            [
                r_idx + 1,
                fa_c1["first_loss"],
                fa_c1["last_loss"],
                fa_c2["first_loss"],
                fa_c2["last_loss"],
                fa["weighted_loss"],
                fa["global_adapter_cid"],
                ff_c1["first_loss"],
                ff_c1["last_loss"],
                ff_c2["first_loss"],
                ff_c2["last_loss"],
                ff["weighted_loss"],
                ff["global_adapter_cid"],
            ]
        )

    # 4. Sharded PDFs & Kubo IPFS CIDs sheet
    pdf_rows: list[list[object]] = [
        [
            "Document ID",
            "Title",
            "Active Shard Owner (Full PDF + Vectors)",
            "Metadata Stub + Backup Peer",
            "Kubo IPFS CIDv1",
            "PDF Binary SHA-256",
            "Content SHA-256 (in PDF /Info)",
            "Chunks SHA-256 (in PDF /Info)",
            "Num Chunks",
            "Page Count",
        ]
    ]
    for m in data["ingested_manifests"]:
        pdf_rows.append(
            [
                m["doc_id"],
                m["title"],
                m["shard_owner"],
                m["metadata_holder"],
                m["ipfs_cid"],
                m["pdf_sha256"],
                m.get("content_sha256", ""),
                m["chunks_sha256"],
                m["num_chunks"],
                m["page_count"],
            ]
        )

    # 5. Related Literature Comparison sheet
    lit_rows: list[list[object]] = [
        [
            "Paper / System",
            "Venue / ID",
            "Year",
            "Permissioned Ledger (Fabric / Raft)",
            "50/50 Sharded PDF RAG + Metadata Stubs",
            "Private Kubo IPFS (LIBP2P_FORCE_PNET=1)",
            "Federated LoRA (FedAvg / FFA-LoRA)",
            "Heterogeneous Edge LLMs (llama.cpp / Ollama)",
        ],
        [
            "FabricFedRAG (Ours)",
            "fabric-fed-rag + softone-rag",
            2026,
            "Yes (Raft + TLS 1.3 mTLS, SHA-256 + CID manifests only)",
            "Yes (50% full PDFs + 50% hash-validated metadata stub PDFs)",
            "Yes (swarm.key + LIBP2P_FORCE_PNET=1 + backup pinning)",
            "Yes (FedAvg-LoRA & FFA-LoRA across 50/50 shards)",
            "Yes (Qwen3.5-4B / Gemma4-26B, CUDA + Apple M4 MPS, GGUF)",
        ],
        [
            "Federated TrustChain (Zuo et al.)",
            "arXiv:2406.04076 (IEEE)",
            2024,
            "Yes (Hyperledger Fabric, uploads full model to chaincode)",
            "No (No RAG, no embeddings, no PDFs)",
            "No",
            "Partial (Unlearning hyperparameter sweep on GPT-2)",
            "No (GPT-2 124M on single 48GB A6000 workstation)",
        ],
        [
            "FD-RAG (Gao et al.)",
            "arXiv:2605.27432",
            2026,
            "No",
            "Partial (Fragmented edge memory hypergraphs)",
            "No",
            "No",
            "Yes (Edge LLM reasoning)",
        ],
        [
            "Blockchain Decentralized RAG",
            "arXiv:2511.07577",
            2025,
            "Yes (Source reliability smart contracts)",
            "No (No 50/50 metadata stub PDFs)",
            "No",
            "No",
            "No",
        ],
        [
            "DRAG (Peer-to-Peer RAG)",
            "arXiv:2505.00443",
            2025,
            "No",
            "Partial (P2P random walk retrieval)",
            "No",
            "No",
            "Yes",
        ],
        [
            "C-FedRAG",
            "arXiv:2412.13163",
            2024,
            "No (Requires hardware TEE enclaves)",
            "Yes (Cross-silo federated RAG)",
            "No",
            "No",
            "No",
        ],
        [
            "FFA-LoRA",
            "arXiv:2403.12313 (ICLR)",
            2024,
            "No",
            "No",
            "No",
            "Yes (Freezes lora_A, averages lora_B)",
            "No",
        ],
        [
            "TAD-LoRA / Dec-LoRA",
            "arXiv:2602.00451 / 2501.15361",
            2026,
            "No",
            "No",
            "No",
            "Yes (Decentralized peer-to-peer LoRA)",
            "Partial",
        ],
    ]

    write_csv(results_dir / "summary_metrics.csv", summary_rows)
    write_csv(results_dir / "per_case_holdout_12.csv", holdout_rows)
    write_csv(results_dir / "lora_ablation_rounds.csv", lora_rows)
    write_csv(results_dir / "sharded_pdfs_ipfs.csv", pdf_rows)
    write_csv(results_dir / "literature_comparison.csv", lit_rows)

    xlsx_path = results_dir / "experimental_results.xlsx"
    write_multi_sheet_xlsx(
        xlsx_path,
        [
            ("Summary_Metrics", summary_rows),
            ("Per_Case_Holdout_12", holdout_rows),
            ("LoRA_Ablation_Rounds", lora_rows),
            ("Sharded_PDFs_IPFS", pdf_rows),
            ("Literature_Comparison", lit_rows),
        ],
    )
    return xlsx_path


if __name__ == "__main__":
    json_path = _ROOT / "results/real_eval_raw.json"
    if json_path.is_file():
        out = export_all_from_json(json_path, _ROOT / "results")
        print(f"Exported Excel workbook to {out}")
