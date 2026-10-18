#!/usr/bin/env bash
# Full pipeline on the PlanarPush reference suite, start to finish.
#
#   bash scripts/run_all.sh [RUN_NAME]
#
# Everything lands in runs/$RUN/{results,checkpoints,logs}. Each step is
# independently re-runnable; later steps read the artefacts of earlier ones,
# so you can iterate on, say, the ablation without recollecting anchors.
set -euo pipefail

RUN="${1:-reference}"
cd "$(dirname "$0")/.."
mkdir -p "runs/$RUN/logs"
LOG="runs/$RUN/logs/run_all.log"
: > "$LOG"

step () {
  local name="$1"; shift
  echo ""                                        | tee -a "$LOG"
  echo "### $name"                               | tee -a "$LOG"
  echo "    $*"                                  | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  echo "    [$name done in $((SECONDS - t0))s]"  | tee -a "$LOG"
}

echo "HALT reference pipeline -- run '$RUN' -- $(date -u +%FT%TZ)" | tee -a "$LOG"

# 0. world model + anchors (Algorithm G.1, lines 1-4)
step "00 collect (ensemble)" python3 scripts/00_collect.py --run "$RUN" \
     --kind ensemble --train-episodes 600 --anchor-episodes 400 --anchors-per-episode 10

# the video-model variant: same machinery, differentiable readout withheld
step "00 collect (video)"    python3 scripts/00_collect.py --run "$RUN" \
     --kind video --train-episodes 600 --anchor-episodes 220 --anchors-per-episode 10

# 1. Algorithm G.1, lines 5-7
step "01 calibrate (ensemble)" python3 scripts/01_calibrate.py --run "$RUN" \
     --kind ensemble --epochs 150
step "01 calibrate (video)"    python3 scripts/01_calibrate.py --run "$RUN" \
     --kind video --epochs 150

# 2. coverage, grounding, silent failure
step "02 evaluate (ensemble)" python3 scripts/02_evaluate.py --run "$RUN" --kind ensemble
step "02 evaluate (video)"    python3 scripts/02_evaluate.py --run "$RUN" --kind video

# 3. PHANTASM six-axis sweep
step "03 phantasm" python3 scripts/03_phantasm.py --run "$RUN" --episodes 80 \
     --anchors-per-episode 6 --severity 2

# 4. delayed-feedback ACI vs static conformal under shift
step "04 aci" python3 scripts/04_aci.py --run "$RUN" --episodes 160 --severity 3

# 5. planner-induced shift and remedies R1/R2
step "05 planner shift" python3 scripts/05_planner_shift.py --run "$RUN" --episodes 60

# 6. feature-family ablation
step "06 ablation" python3 scripts/06_ablation.py --run "$RUN" --epochs 120

# 7. closed-loop deployment vs fixed horizons
step "07 deploy" python3 scripts/07_deploy.py --run "$RUN" --episodes 25

# 8. assemble the summary
step "08 tables" python3 scripts/08_tables.py --run "$RUN"

echo "" | tee -a "$LOG"
echo "Finished. Artefacts in runs/$RUN/" | tee -a "$LOG"
ls -1 "runs/$RUN/results" | sed 's/^/  /' | tee -a "$LOG"
