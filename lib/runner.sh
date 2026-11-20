#!/usr/bin/env bash
# Minimal job runner inside one srun step (avoids per-step port limits on shared nodes).
# Drop executable scripts into $Q/gpu/ or $Q/cpu/; each runs once. GPU jobs get CUDA_VISIBLE_DEVICES of a free slot.
# Stop: touch $Q/STOP.   Usage: srun --jobid=J --overlap -n1 --gpus=N -c C bash lib/runner.sh QDIR NGPU NCPUJOBS
Q=${1:?queue dir}; NGPU=${2:-2}; NCPU=${3:-2}
mkdir -p "$Q"/{gpu,cpu,running,done,failed}
declare -A GPID CPID
while [ ! -e "$Q/STOP" ]; do
  for g in $(seq 0 $((NGPU - 1))); do
    p=${GPID[$g]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      f=$(ls -1 "$Q"/gpu/*.sh 2>/dev/null | head -1)
      if [ -n "$f" ]; then
        b=$(basename "$f"); mv "$f" "$Q/running/$b" 2>/dev/null || continue
        ( CUDA_VISIBLE_DEVICES=$g bash "$Q/running/$b" > "$Q/running/$b.log" 2>&1 \
            && mv "$Q/running/$b" "$Q/running/$b.log" "$Q/done/" \
            || mv "$Q/running/$b" "$Q/running/$b.log" "$Q/failed/" ) &
        GPID[$g]=$!
      fi
    fi
  done
  for c in $(seq 0 $((NCPU - 1))); do
    p=${CPID[$c]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      f=$(ls -1 "$Q"/cpu/*.sh 2>/dev/null | head -1)
      if [ -n "$f" ]; then
        b=$(basename "$f"); mv "$f" "$Q/running/$b" 2>/dev/null || continue
        ( CUDA_VISIBLE_DEVICES= bash "$Q/running/$b" > "$Q/running/$b.log" 2>&1 \
            && mv "$Q/running/$b" "$Q/running/$b.log" "$Q/done/" \
            || mv "$Q/running/$b" "$Q/running/$b.log" "$Q/failed/" ) &
        CPID[$c]=$!
      fi
    fi
  done
  sleep 10
done
wait
