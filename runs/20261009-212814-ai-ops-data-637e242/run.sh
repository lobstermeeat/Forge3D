#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-data: Phase 9, how many clean 3D models we may train on. ops/data_audit.py reads TexVerse's metadata
# (licence, PBR, categories) and Objaverse++'s quality scores from Hugging Face, on this runner: no Modal, no GPU,
# metadata only. The counts go to ops-out/data-summary.json; the uid lists stay in ops-out/private/data/.
set -euo pipefail
mkdir -p ops-out/private
pip install -q "huggingface_hub>=0.34,<2" "ijson>=3.3" >/dev/null
status=0
timeout 9000 python3 ops/data_audit.py ops-out/private/data 2>&1 | tail -40 || status=$?
cp ops-out/private/data/summary.json ops-out/data-summary.json 2>/dev/null || true
du -sh ops-out/private/data 2>/dev/null || true
exit $status
