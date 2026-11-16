#!/bin/sh
set -e

cp /certs/swarm.key /data/ipfs/swarm.key
chmod 600 /data/ipfs/swarm.key

ipfs bootstrap rm --all
ipfs config --json AutoConf.Enabled false
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
ipfs config --json Routing.Type '"dht"'
ipfs config --json Discovery.MDNS.Enabled true
