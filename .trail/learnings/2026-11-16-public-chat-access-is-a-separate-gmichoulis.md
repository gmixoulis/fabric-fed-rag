---
title: "Public chat access is a separate port with a login, never a rule on the open local port"
author: Gmichoulis
date: 2026-11-16
branch: main
cost: { minutes: 139, tokens: 739883 }
id: l-7e1c874a
scope: repo
status: active
src: agent
touches:
  - ".python-version"
  - AGENTS.md
  - README.md
  - TUTORIAL.md
  - deploy/README.md
  - deploy/macos/local.llama.gpulimit.plist
  - deploy/macos/local.llama.rpc.plist
  - deploy/macos/start-rpc.sh
  - deploy/systemd/llama-split.service
  - deploy/systemd/softone-chat-public.service
---

## Decided
Two chat services share the code: one on the local port without a login, one on another port with the password login. Only the second may be a reverse-proxy or Funnel target.

## Tried and ruled out
One port that skips the login for local addresses and requires it for the public name: a local reverse proxy also connects from 127.0.0.1 and may rewrite the Host header, so public visitors could be treated as local.

## Watch out
- The server refuses to start with public host names and no password hash.
- The failed-login limit is global: a stranger can lock the owner out for a minute at a time, but cannot get in.
- Sessions are in memory; a restart signs everyone out.
- The password hash and allowed hosts live outside the repository, mode 0600.

In the code: src/softone_rag/auth.py, src/softone_rag/web.py (make_server password_hash, allowed_hosts), deploy/systemd/softone-chat.service (8877), softone-chat-public.service (8878).
