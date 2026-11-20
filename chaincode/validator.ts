/**
 * TypeScript Chaincode Validator for Hyperledger Fabric 2.x / Raft Ledger
 *
 * Validates minimal on-chain commitments for:
 * 1. DOC_MANIFEST (hash-sharded PDF & chunk manifest with SHA-256 hashes)
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

export interface ValidatorConfig {
  /** Full raft membership, in node ids. Empty/omitted keeps the legacy permissive mode. */
  members?: string[];
  /** Minimum distinct submitters accepted by LORA_ROUND_FINALIZE. Default: majority of members. */
  loraQuorum?: number;
}

const isSha256 = (value: string): boolean => /^[a-f0-9]{64}$/i.test(value);

const majority = (nodeCount: number): number => {
  if (nodeCount <= 0) {
    throw new Error("a raft membership needs at least one node");
  }
  return Math.floor(nodeCount / 2) + 1;
};

export function validateTransaction(
  txType: TxType,
  payload: unknown,
  documents: Record<string, DocManifest>,
  clientSubs: Record<number, Record<string, LoRAClientSubmission>>,
  config: ValidatorConfig = {},
): void {
  const members = (config.members ?? []).filter(Boolean);
  const loraQuorum = config.loraQuorum ?? (members.length > 0 ? majority(members.length) : 1);
  if (loraQuorum <= 0) {
    throw new Error("loraQuorum must be positive");
  }
  if (members.length > 0 && loraQuorum > members.length) {
    throw new Error("loraQuorum cannot exceed the number of configured members");
  }
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
    if (members.length > 0 && !members.includes(sub.node_id)) {
      throw new Error(`LORA_CLIENT_SUBMIT from non-member ${sub.node_id}`);
    }
    const recorded = clientSubs[sub.round_id]?.[sub.node_id];
    if (recorded && recorded.adapter_sha256 !== sub.adapter_sha256) {
      throw new Error(
        `Conflicting duplicate LORA_CLIENT_SUBMIT for ${sub.node_id} in round ${sub.round_id}`,
      );
    }
    return;
  }

  if (txType === "LORA_ROUND_FINALIZE") {
    const fin = payload as LoRARoundFinalized;
    if (fin.round_id <= 0 || !isSha256(fin.global_adapter_sha256)) {
      throw new Error("Invalid LORA_ROUND_FINALIZE fields");
    }
    const entries = Object.entries(fin.participant_hashes ?? {});
    if (entries.length === 0) {
      throw new Error(`LORA_ROUND_FINALIZE for round ${fin.round_id} requires participant_hashes`);
    }

    const declared = (fin as LoRARoundFinalized & { participants?: unknown }).participants;
    if (declared !== undefined) {
      if (!Array.isArray(declared) || declared.some((id) => typeof id !== "string" || id.length === 0)) {
        throw new Error("LORA_ROUND_FINALIZE participants must be a list of node ids");
      }
      const participantIds = declared as string[];
      if (new Set(participantIds).size !== participantIds.length) {
        throw new Error(`Duplicate submitter in LORA_ROUND_FINALIZE for round ${fin.round_id}`);
      }
      const declaredSorted = [...participantIds].sort();
      const hashedSorted = entries.map(([nodeId]) => nodeId).sort();
      if (
        declaredSorted.length !== hashedSorted.length ||
        declaredSorted.some((id, index) => id !== hashedSorted[index])
      ) {
        throw new Error(
          `LORA_ROUND_FINALIZE participants do not match participant_hashes for round ${fin.round_id}`,
        );
      }
    }

    const roundSubs = clientSubs[fin.round_id] ?? {};
    for (const [nodeId, expectedHash] of entries) {
      if (!isSha256(expectedHash)) {
        throw new Error(`Invalid participant hash for ${nodeId} in round ${fin.round_id}`);
      }
      if (members.length > 0 && !members.includes(nodeId)) {
        throw new Error(`Non-member submitter ${nodeId} in LORA_ROUND_FINALIZE for round ${fin.round_id}`);
      }
      const committed = roundSubs[nodeId];
      if (!committed) {
        throw new Error(`Missing LORA_CLIENT_SUBMIT for ${nodeId} in round ${fin.round_id}`);
      }
      if (committed.adapter_sha256 !== expectedHash) {
        throw new Error(`Adapter hash mismatch for ${nodeId} in round ${fin.round_id}`);
      }
    }
    if (entries.length < loraQuorum) {
      throw new Error(
        `LORA_ROUND_FINALIZE for round ${fin.round_id} requires submissions from ${loraQuorum} member(s), got ${entries.length}`,
      );
    }
    return;
  }

  throw new Error(`Unsupported transaction type: ${txType}`);
}
