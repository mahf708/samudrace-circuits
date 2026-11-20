#!/usr/bin/env bash
# Populate ./data with the captured states and truth extracts used by the scripts.
#  - On Perlmutter: read-only symlinks to the shared copy (no download).
#  - Elsewhere: download from the HF dataset mahf708/samudrace-circuits (~21 GB; pass subfolders to limit,
#    e.g. `bash fetch_data.sh states/t2045 f1`).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED=/pscratch/sd/m/mahf708/hackathon-circuits/data
ITEMS=("${@:-states/t1995 states/t2045 states/atm12 states/atm12m5 states/ic1985 f1 rollout1y}")
ITEMS=(${ITEMS[@]})
mkdir -p "$ROOT/data/states"
if [ -r "$SHARED/states/t2045/meta.npz" ] && [ "$(cd "$ROOT/data" && pwd -P)" != "$SHARED" ]; then
  for it in "${ITEMS[@]}"; do
    [ -e "$ROOT/data/$it" ] || ln -s "$SHARED/$it" "$ROOT/data/$it"
    echo "linked data/$it -> $SHARED/$it"
  done
else
  HF=${HF:-hf}
  for it in "${ITEMS[@]}"; do
    "$HF" download mahf708/samudrace-circuits --repo-type dataset --include "$it/*" --local-dir "$ROOT/data"
  done
fi
echo "data ready under $ROOT/data"
