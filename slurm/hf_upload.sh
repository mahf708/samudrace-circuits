#!/bin/bash
# Upload derived data products to HF dataset mahf708/samudrace-circuits (public).
export HF_HOME=$PSCRATCH/hf; cd /pscratch/sd/m/mahf708/hackathon-circuits/data
EXC=(--exclude "forcing/*" --exclude "**/forcing/*" --exclude "*diagnostics*.nc" --exclude "**/*diagnostics*.nc" --exclude "*.log")
for d in states/t1995 states/t2045 states/atm12 states/atm12m5 states/ic1985 f1; do
  echo "== $d $(date)"; hf upload mahf708/samudrace-circuits $d $d --repo-type dataset "${EXC[@]}" --commit-message "data: $d" 2>&1 | tail -2
done
echo "ALL DONE $(date)"
