"""Offline checks for the three-organisation certificate and connection profile (WP1)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fedrag.certgen import generate_certs

NODE_ORGS = {
    "node1-maestro": "MaestroOrg",
    "node2-peer": "Peer2Org",
    "node3-peer": "Peer3Org",
}


def _openssl_x509(cert: Path, *flags: str) -> str:
    return subprocess.run(
        ["openssl", "x509", "-in", str(cert), "-noout", *flags],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_certgen_issues_third_org_certificate_with_rfc5280_extensions(tmp_path):
    certs = tmp_path / "certs"
    generate_certs(certs, list(NODE_ORGS), ["127.0.0.1"])

    assert (certs / "ca.crt").is_file()
    ca_text = _openssl_x509(certs / "ca.crt", "-text")
    assert "X509v3 Subject Key Identifier" in ca_text
    assert "X509v3 Authority Key Identifier" in ca_text

    for node, org in NODE_ORGS.items():
        cert = certs / f"{node}.crt"
        key = certs / f"{node}.key"
        assert cert.is_file() and key.is_file()
        assert key.stat().st_mode & 0o777 == 0o600

        subject = _openssl_x509(cert, "-subject", "-nameopt", "RFC2253")
        assert f"CN={node}" in subject
        assert f"O={org}" in subject

        text = _openssl_x509(cert, "-text")
        assert "X509v3 Subject Key Identifier" in text
        assert "X509v3 Authority Key Identifier" in text
        assert f"DNS:{node}" in text
        assert "TLS Web Server Authentication" in text
        assert "TLS Web Client Authentication" in text


def test_private_keys_stay_out_of_git():
    root = Path(__file__).resolve().parents[1]
    ignored = [line.strip() for line in (root / ".gitignore").read_text().splitlines() if line.strip()]
    assert "certs/" in ignored
    # The generated key paths in the connection profile must live under that ignored directory.
    profile = json.loads((root / "connection-profile.json").read_text())
    for peer in profile["peers"].values():
        assert peer["clientCert"]["keyPath"].startswith("./certs/")


def test_connection_profile_declares_peer3org_and_three_member_raft():
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "connection-profile.json").read_text())

    assert profile["organizations"]["Peer3Org"]["mspid"] == "Peer3OrgMSP"
    assert profile["organizations"]["Peer3Org"]["peers"] == ["node3-peer"]

    channel = profile["channels"]["rag-fed-channel"]
    assert "node3-peer" in channel["peers"]
    assert "node3-peer" in channel["orderers"]
    assert "node3-peer" in profile["peers"]
    assert profile["peers"]["node3-peer"]["clientCert"] == {
        "certPath": "./certs/node3-peer.crt",
        "keyPath": "./certs/node3-peer.key",
    }

    raft = profile["raft"]
    assert raft["members"] == ["node1-maestro", "node2-peer", "node3-peer"]
    assert raft["memberUrls"]["node3-peer"] == "https://node3-peer:7050"
    assert raft["loraQuorum"] == 2
