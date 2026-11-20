"""Kubo IPFS Private Swarm Client & Deterministic CIDv1 Offline Store.

Connects to the local Kubo HTTP RPC API (`http://localhost:5001/api/v0/...`)
running in Private Swarm mode (`LIBP2P_FORCE_PNET=1` + `swarm.key`) to:
- Add and pin shard PDFs and LoRA adapter weights (`POST /api/v0/add?pin=true&cid-version=1`)
- Replicate and pin backup CIDs across peers (`POST /api/v0/pin/add?arg=<CID>`)
- Retrieve content by CID (`POST /api/v0/cat?arg=<CID>`)
- Connect private swarm peers (`POST /api/v0/swarm/connect?arg=<multiaddr>`)

For offline unit tests (`pytest -q`), `LocalCIDStore` computes the exact RFC-compliant
CIDv1 raw SHA-256 multihash (`bafkrei...`) in memory without network calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse
import urllib.request
import uuid


def compute_cidv1_raw(data: bytes) -> str:
    """Compute a deterministic CIDv1 (raw codec 0x55, sha2-256 0x12, 32 bytes 0x20) in base32 ('bafkrei...')."""
    digest = hashlib.sha256(data).digest()
    raw_cid = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    b32 = base64.b32encode(raw_cid).decode("ascii").lower().rstrip("=")
    return "b" + b32


class LocalCIDStore:
    """Offline, zero-network CIDv1 store for unit tests and fallback mode."""

    mode = "local-cidv1"
    _shared_blobs: dict[str, bytes] = {}

    def __init__(self, node_name: str = "local") -> None:
        self.node_name = node_name
        self.pins: set[str] = set()

    def add_bytes(self, data: bytes, filename: str = "file.bin", pin: bool = True) -> str:
        cid = compute_cidv1_raw(data)
        self._shared_blobs[cid] = bytes(data)
        if pin:
            self.pins.add(cid)
        return cid

    def cat_bytes(self, cid: str) -> bytes:
        if cid not in self._shared_blobs:
            raise KeyError(f"CID {cid} not found in LocalCIDStore")
        return self._shared_blobs[cid]

    def pin_add(self, cid: str) -> list[str]:
        if cid not in self._shared_blobs:
            raise KeyError(f"Cannot pin unknown CID {cid}")
        self.pins.add(cid)
        return [cid]

    def pin_ls(self) -> list[str]:
        return sorted(self.pins)


class KuboIPFSClient:
    """HTTP RPC client for a local Kubo IPFS daemon (`http://ipfs-node:5001`)."""

    mode = "kubo-pnet"

    def __init__(self, api_url: str = "http://localhost:5001", timeout: float = 30.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, params: dict[str, str] | None = None) -> bytes:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.api_url}/api/v0/{path.lstrip('/')}{query}"
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def id_info(self) -> dict:
        return json.loads(self._post("id").decode())

    def swarm_peers(self) -> list[str]:
        data = json.loads(self._post("swarm/peers").decode())
        peers = data.get("Peers") or []
        return [f"{p.get('Addr')}/p2p/{p.get('Peer')}" for p in peers]

    def swarm_connect(self, multiaddr: str) -> dict:
        return json.loads(self._post("swarm/connect", {"arg": multiaddr}).decode())

    def add_bytes(self, data: bytes, filename: str = "blob.bin", pin: bool = True) -> str:
        boundary = f"----FabricFedRAGBoundary{uuid.uuid4().hex}"
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        footer = f"\r\n--{boundary}--\r\n".encode()
        body = header + data + footer

        params = urllib.parse.urlencode(
            {"pin": "true" if pin else "false", "cid-version": "1", "quieter": "true"}
        )
        url = f"{self.api_url}/api/v0/add?{params}"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            lines = [line for line in resp.read().decode().splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        return payload["Hash"]

    def cat_bytes(self, cid: str) -> bytes:
        return self._post("cat", {"arg": cid})

    def pin_add(self, cid: str) -> list[str]:
        res = json.loads(self._post("pin/add", {"arg": cid}).decode())
        return res.get("Pins") or [cid]

    def pin_ls(self) -> list[str]:
        res = json.loads(self._post("pin/ls", {"type": "recursive"}).decode())
        keys = res.get("Keys") or {}
        return sorted(keys.keys())
