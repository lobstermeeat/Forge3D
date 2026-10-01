#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Experiments run on their own ai-ops-<name> branches, each with its own ops/run.sh, side by side.
# They use `modal run` with their own app names, never `modal deploy workers/modal_app.py`, which
# replaces the production app orainge-ai.
set -euo pipefail

echo "== Relay check: the apps on Modal"
modal app list 2>&1 | head -20
