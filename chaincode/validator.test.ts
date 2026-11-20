/**
 * Offline checks for the TypeScript chaincode twin.
 * Run with: node --experimental-strip-types chaincode/validator.test.ts
 */
import assert from "node:assert/strict";

import {
  type DocManifest,
  type LoRAClientSubmission,
  type LoRARoundFinalized,
  validateTransaction,
} from "./validator.ts";

const A = "a".repeat(64);
const B = "b".repeat(64);
const C = "c".repeat(64);
const D = "d".repeat(64);
const MEMBERS = ["node1-maestro", "node2-peer", "node3-peer"];

const documents: Record<string, DocManifest> = {};
const submissions: Record<number, Record<string, LoRAClientSubmission>> = {};

const submit = (round: number, node: string, sha: string): LoRAClientSubmission => ({
  round_id: round,
  node_id: node,
  shard_id: "shard-1",
  adapter_sha256: sha,
  num_examples: 5,
  first_loss: 1,
  last_loss: 0.5,
});

const finalize = (round: number, participants: string[], hashes: Record<string, string>): LoRARoundFinalized => ({
  round_id: round,
  global_adapter_sha256: D,
  participant_hashes: hashes,
  total_examples: 10,
  weighted_loss: 0.5,
  ...({ participants } as object),
});

for (const [node, sha] of [["node1-maestro", A], ["node2-peer", B], ["node3-peer", C]] as const) {
  submissions[1] = { ...(submissions[1] ?? {}), [node]: submit(1, node, sha) };
}

// Two of three members is the configured default majority.
validateTransaction(
  "LORA_ROUND_FINALIZE",
  finalize(1, ["node1-maestro", "node2-peer"], { "node1-maestro": A, "node2-peer": B }),
  documents,
  submissions,
  { members: MEMBERS },
);

// One of three is below quorum.
assert.throws(
  () =>
    validateTransaction(
      "LORA_ROUND_FINALIZE",
      finalize(1, ["node1-maestro"], { "node1-maestro": A }),
      documents,
      submissions,
      { members: MEMBERS },
    ),
  /requires submissions from 2/,
);

// A non-member cannot submit or appear in a finalize.
assert.throws(
  () =>
    validateTransaction(
      "LORA_CLIENT_SUBMIT",
      submit(1, "node4-outsider", A),
      documents,
      submissions,
      { members: MEMBERS },
    ),
  /non-member/,
);
assert.throws(
  () =>
    validateTransaction(
      "LORA_ROUND_FINALIZE",
      finalize(
        1,
        ["node1-maestro", "node4-outsider"],
        { "node1-maestro": A, "node4-outsider": B },
      ),
      documents,
      submissions,
      { members: MEMBERS },
    ),
  /Non-member submitter/,
);

// A duplicate submitter is rejected.
assert.throws(
  () =>
    validateTransaction(
      "LORA_ROUND_FINALIZE",
      finalize(1, ["node1-maestro", "node1-maestro"], { "node1-maestro": A }),
      documents,
      submissions,
      { members: MEMBERS },
    ),
  /Duplicate submitter/,
);

// A conflicting second submission for the same round and node is rejected.
assert.throws(
  () =>
    validateTransaction(
      "LORA_CLIENT_SUBMIT",
      submit(1, "node1-maestro", B),
      documents,
      submissions,
      { members: MEMBERS },
    ),
  /Conflicting duplicate/,
);

// An explicit quorum is honoured, and raising it above the membership is refused.
assert.throws(
  () =>
    validateTransaction(
      "LORA_ROUND_FINALIZE",
      finalize(1, ["node1-maestro", "node2-peer"], { "node1-maestro": A, "node2-peer": B }),
      documents,
      submissions,
      { members: MEMBERS, loraQuorum: 3 },
    ),
  /requires submissions from 3/,
);
assert.throws(
  () => validateTransaction("LORA_ROUND_FINALIZE", finalize(1, [], {}), documents, submissions, {
    members: MEMBERS,
    loraQuorum: 4,
  }),
  /cannot exceed/,
);

// Legacy callers without membership configuration keep accepting one valid submitter.
validateTransaction(
  "LORA_ROUND_FINALIZE",
  finalize(1, ["node1-maestro"], { "node1-maestro": A }),
  documents,
  submissions,
);

console.log("chaincode validator checks passed");
