#!/usr/bin/env bash
# Rebuild the hackathon-circuits environment from scratch.
#   bash setup_env.sh            # creates ./.venv (Python 3.12), ./ace-tag, installs everything editable
# Everything lives under this directory; caches go to $PSCRATCH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PSCRATCH/.cache/uv}"
ACE_REPO="${ACE_REPO:-}"                         # optional: a local ace clone that has the tag
TAG=e3smv3-lr-hist-samudrace-rc1                # commit 1c29e2d9b29a876bd9c469593baa8b63e4256370
TORCH_VERSION=2.14.1                            # matches the release's uv.lock (cu130 wheels on PyPI)

# 1. ace at the tag, as a plain tree (git archive), installed editable
if [ ! -d ace-tag ]; then
  if [ -n "$ACE_REPO" ] && git -C "$ACE_REPO" rev-parse -q --verify "$TAG^{commit}" >/dev/null; then
    mkdir ace-tag
    git -C "$ACE_REPO" archive "$TAG" | tar -x -C ace-tag
    git -C "$ACE_REPO" rev-parse "$TAG^{commit}" > ace-tag/.tag_commit
  else  # shallow clone of the tag from GitHub, then drop .git (same tree as git archive)
    git clone -q --depth 1 --branch "$TAG" https://github.com/E3SM-Project/ace.git ace-tag
    git -C ace-tag rev-parse HEAD > ace-tag/.tag_commit && rm -rf ace-tag/.git
  fi
fi

# 2. exege: circuits + laws + the samudra adapter (branch of PR mahf708/exege#12; use main once merged)
EXEGE_BRANCH="${EXEGE_BRANCH:-mahf708/samudra-adapter}"
if [ ! -d exege ]; then
  git clone -q https://github.com/mahf708/exege.git exege
  git -C exege checkout -q -b "$EXEGE_BRANCH" "origin/$EXEGE_BRANCH"
fi

# 3. venv
[ -d .venv ] || uv venv --python 3.12 .venv
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" "torch==$TORCH_VERSION"
uv pip install --python "$PY" -e ./ace-tag
uv pip install --python "$PY" -e "./exege[laws,figures,hf,dev]" 2>/dev/null \
  || uv pip install --python "$PY" -e ./exege
uv pip install --python "$PY" -e ./lib                         # hc package; registers the samudra-replay entry point
uv pip install --python "$PY" scipy netCDF4 h5netcdf matplotlib pandas pytest ruff

# 4. check
"$PY" - <<'EOF'
import torch, fme, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("fme", fme.__file__)
eps = [e.name for e in md.entry_points(group="exege.adapters")]
print("exege.adapters entry points:", eps)
assert "samudra-replay" in eps
EOF
