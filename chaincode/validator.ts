/**
 * TypeScript Chaincode Validator for Hyperledger Fabric 2.x / Raft Ledger
 *
 * Validates minimal on-chain commitments for:
 * 1. DOC_MANIFEST (50/50 PDF & chunk shard manifest with SHA-256 hashes)
 * 2. BACKUP_CHECK (Cross-hardware backup replica SHA-256 verification)
 * 3. LORA_CLIENT_SUBMIT (Peer LoRA adapter SHA-256 & loss metrics)
 * 4. LORA_ROUND_FINALIZE (Maestro FedAvg / FFA-LoRA global adapter SHA-256 verification)
 */

export type TxType =
  | "GENESIS"
  | "DOC_MANIFEST"
  | "BACKUP_CHECK"
  | "LORA_CLIENT_SUBMIT"
  | "LORA_ROUND_FINALIZE";

export interface DocManifest {
  doc_id: string;
  title: string;
  shard_owner: string;
  metadata_holder: string;
  ipfs_cid?: string;
  pdf_sha256: string;
  chunks_sha256: string;
  num_chunks: number;
  page_count: number;
}

export interface BackupAttestation {
  doc_id: string;
  source_node: string;
  backup_node: string;
  ipfs_cid?: string;
  computed_sha256: string;
  verified: boolean;
}

export interface LoRAClientSubmission {
  round_id: number;
  node_id: string;
  shard_id: string;
  lora_mode?: "fedavg" | "ffa" | "dora";
  adapter_cid?: string;
  adapter_sha256: string;
  num_examples: number;
  first_loss: number;
  last_loss: number;
}

export interface LoRARoundFinalized {
  round_id: number;
  lora_mode?: "fedavg" | "ffa" | "dora";
  global_adapter_cid?: string;
  global_adapter_sha256: string;
  participant_hashes: Record<string, string>;
  total_examples: number;
  weighted_loss: number;
}

const isSha256 = (value: string): boolean => /^[a-f0-9]{64}$/i.test(value);

export function validateTransaction(
  txType: TxType,
  payload: unknown,
  documents: Record<string, DocManifest>,
  clientSubs: Record<number, Record<string, LoRAClientSubmission>>,
): void {
  if (txType === "DOC_MANIFEST") {
    const doc = payload as DocManifest;
    if (!doc.doc_id || !isSha256(doc.pdf_sha256) || !isSha256(doc.chunks_sha256)) {
      throw new Error("DOC_MANIFEST requires doc_id and valid 64-char SHA-256 hashes");
    }
    if (!doc.shard_owner || !doc.metadata_holder) {
      throw new Error("DOC_MANIFEST requires shard_owner and metadata_holder");
    }
    const recorded = documents[doc.doc_id];
    if (recorded) {
      const fields: (keyof DocManifest)[] = [
        "doc_id",
        "title",
        "shard_owner",
        "metadata_holder",
        "ipfs_cid",
        "pdf_sha256",
        "chunks_sha256",
        "num_chunks",
        "page_count",
      ];
      if (fields.some((field) => doc[field] !== recorded[field])) {
        throw new Error(`DOC_MANIFEST for ${doc.doc_id} conflicts with the recorded manifest`);
      }
    }
    return;
  }

  if (txType === "BACKUP_CHECK") {
    const backup = payload as BackupAttestation;
    const expectedDoc = documents[backup.doc_id];
    if (!expectedDoc) {
      throw new Error(`Unknown doc_id ${backup.doc_id} for BACKUP_CHECK`);
    }
    if (backup.computed_sha256 !== expectedDoc.pdf_sha256) {
      throw new Error(
        `Backup checksum mismatch for ${backup.doc_id}: ${backup.computed_sha256} !== ${expectedDoc.pdf_sha256}`,
      );
    }
    if (backup.ipfs_cid && expectedDoc.ipfs_cid && backup.ipfs_cid !== expectedDoc.ipfs_cid) {
      throw new Error(
        `Backup IPFS CID mismatch for ${backup.doc_id}: ${backup.ipfs_cid} !== ${expectedDoc.ipfs_cid}`,
      );
    }
    return;
  }

  if (txType === "LORA_CLIENT_SUBMIT") {
    const sub = payload as LoRAClientSubmission;
    if (sub.round_id <= 0 || !sub.node_id || !isSha256(sub.adapter_sha256) || sub.num_examples <= 0) {
      throw new Error("Invalid LORA_CLIENT_SUBMIT fields");
    }
    return;
  }

  if (txType === "LORA_ROUND_FINALIZE") {
    const fin = payload as LoRARoundFinalized;
    if (fin.round_id <= 0 || !isSha256(fin.global_adapter_sha256)) {
      throw new Error("Invalid LORA_ROUND_FINALIZE fields");
    }
    const participants = Object.entries(fin.participant_hashes ?? {});
    if (participants.length === 0) {
      throw new Error(`LORA_ROUND_FINALIZE for round ${fin.round_id} requires participant_hashes`);
    }
    const roundSubs = clientSubs[fin.round_id] ?? {};
    for (const [nodeId, expectedHash] of participants) {
      if (!isSha256(expectedHash)) {
        throw new Error(`Invalid participant hash for ${nodeId} in round ${fin.round_id}`);
      }
      const committed = roundSubs[nodeId];
      if (!committed) {
        throw new Error(`Missing LORA_CLIENT_SUBMIT for ${nodeId} in round ${fin.round_id}`);
      }
      if (committed.adapter_sha256 !== expectedHash) {
        throw new Error(`Adapter hash mismatch for ${nodeId} in round ${fin.round_id}`);
      }
    }
    return;
  }

  throw new Error(`Unsupported transaction type: ${txType}`);
}
