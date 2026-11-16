---
title: The split model server needs two memory flags or the low-memory guard kills it mid-run
author: Gmichoulis
date: 2026-11-16
branch: main
cost: { minutes: 71, tokens: 447184 }
id: l-e79886ba
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
Run the split llama-server with the model not memory-mapped and with a small prompt cache. On this PC the host swap is full, and earlyoom sends SIGTERM to the largest process when RAM and swap both drop under 10%.

## Tried and ruled out
- Blaming the application or the evaluation: three evaluation runs died at the same case, which looked like a bad prompt. It was memory growth reaching the limit at the same point each time.
- `--no-mmap`: not a valid flag in the pinned build. The unit restart-looped and the chat was down for about 30 minutes.
- Removing the memory map alone: resident memory fell from 7 GB to 1.7 GB after load, then grew back to 7.3 GB during the run.

## Watch out
- The prompt cache defaults to 8192 MiB of host RAM, more than the unit's 6 GB cap.
- Check `llama-server --help` of the pinned build before editing the unit.
- The Mac side is separate: its limit is GPU memory, raised to 14 GB at boot.

In the code: deploy/systemd/llama-split.service (`--load-mode none --cache-ram 1024`, MemoryMax=6G); peak after the fix 4.5 GB over a 25-case run. Diagnose with `journalctl | grep earlyoom`.
