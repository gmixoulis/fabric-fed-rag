"""Generate Fabric-style Root CA and X.509 mTLS certificates for nodes using openssl."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def generate_certs(out_dir: Path, nodes: list[str], extra_ips: list[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ca_key = out_dir / "ca.key"
    ca_crt = out_dir / "ca.crt"

    subprocess.run(
        ["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(ca_key)],
        check=True,
        capture_output=True,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as ca_cnf:
        ca_cnf.write(
            "[req]\n"
            "distinguished_name=dn\n"
            "x509_extensions=v3_ca\n"
            "prompt=no\n"
            "[dn]\n"
            "O=FabricFedRAG Consortium CA\n"
            "CN=FabricFedRAG-Root-CA\n"
            "[v3_ca]\n"
            "basicConstraints=critical,CA:TRUE\n"
            "keyUsage=critical,keyCertSign,cRLSign\n"
            "subjectKeyIdentifier=hash\n"
            "authorityKeyIdentifier=keyid:always,issuer\n"
        )
        ca_cnf_path = Path(ca_cnf.name)

    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-sha256",
                "-key",
                str(ca_key),
                "-out",
                str(ca_crt),
                "-days",
                "365",
                "-config",
                str(ca_cnf_path),
                "-extensions",
                "v3_ca",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        ca_cnf_path.unlink(missing_ok=True)
    ca_key.chmod(0o600)
    ca_crt.chmod(0o644)

    for node in nodes:
        node_key = out_dir / f"{node}.key"
        node_csr = out_dir / f"{node}.csr"
        node_crt = out_dir / f"{node}.crt"

        subprocess.run(
            ["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(node_key)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-key",
                str(node_key),
                "-out",
                str(node_csr),
                "-subj",
                f"/O=FabricFedRAG Peer MSP/CN={node}",
            ],
            check=True,
            capture_output=True,
        )

        dns_entries = [f"DNS:{node}", "DNS:localhost", "DNS:host.docker.internal"]
        ip_entries = [f"IP:{ip}" for ip in extra_ips if ip]
        san_line = "subjectAltName=" + ",".join(dns_entries + ip_entries)

        with tempfile.NamedTemporaryFile("w", suffix=".ext", delete=False) as ext_file:
            ext_file.write(
                f"{san_line}\n"
                "basicConstraints=CA:FALSE\n"
                "keyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth,clientAuth\n"
                "subjectKeyIdentifier=hash\n"
                "authorityKeyIdentifier=keyid,issuer\n"
            )
            ext_path = Path(ext_file.name)

        try:
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-req",
                    "-in",
                    str(node_csr),
                    "-CA",
                    str(ca_crt),
                    "-CAkey",
                    str(ca_key),
                    "-CAcreateserial",
                    "-out",
                    str(node_crt),
                    "-days",
                    "365",
                    "-sha256",
                    "-extfile",
                    str(ext_path),
                ],
                check=True,
                capture_output=True,
            )
        finally:
            ext_path.unlink(missing_ok=True)
            node_csr.unlink(missing_ok=True)

        node_key.chmod(0o600)
        node_crt.chmod(0o644)

    srl = out_dir / "ca.srl"
    srl.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FabricFedRAG X.509 mTLS certificates")
    parser.add_argument("--out", default="./certs", help="Output directory")
    parser.add_argument("--nodes", default="node1-maestro,node2-peer", help="Comma-separated node names")
    parser.add_argument("--ips", default="127.0.0.1", help="Comma-separated IP SANs")
    args = parser.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    ips = [ip.strip() for ip in args.ips.split(",") if ip.strip()]
    generate_certs(Path(args.out), nodes, ips)
    print(f"Generated Fabric-style mTLS CA and node certificates in {args.out}")


if __name__ == "__main__":
    main()
