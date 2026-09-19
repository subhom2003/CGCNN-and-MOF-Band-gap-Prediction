#!/usr/bin/env bash
# Full pipeline: clean -> EDA -> graphs -> splits -> train -> evaluate
#                -> baselines -> interpret -> predict(test ids)
#
# Usage:
#   bash run_pipeline.sh                                  # full run
#   bash run_pipeline.sh --smoke_test true --epochs 5     # quick dry run
#   SKIP_BASELINES=1 SKIP_INTERPRET=1 bash run_pipeline.sh
#
# Any extra args are forwarded to EVERY stage (they are CFG fields), e.g.:
#   bash run_pipeline.sh --cutoff 5.0 --batch_size 32 --seed 7
set -euo pipefail
cd "$(dirname "$0")"

# use project venv if it exists (on Colab there is none -> system python)
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

EXTRA_ARGS=("$@")
mkdir -p logs

run_stage () {
    local name="$1"; shift
    echo ""
    echo "=================================================================="
    echo ">>> STAGE: $name"
    echo "=================================================================="
    python -m "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "logs/${name}.log"
}

run_stage 01_data_loading  src.data_loading
run_stage 02_eda           src.eda
run_stage 03_graph_builder src.graph_builder
run_stage 04_preprocessing src.preprocessing
run_stage 05_train         src.train
run_stage 06_evaluate      src.evaluate

if [ "${SKIP_BASELINES:-0}" != "1" ]; then
    run_stage 07_baselines src.baselines
fi

if [ "${SKIP_INTERPRET:-0}" != "1" ]; then
    run_stage 08_interpret src.interpret
fi

run_stage 09_predict src.predict

echo ""
echo "Pipeline finished."
echo "  Model        : outputs/models/best_model.pt"
echo "  Metrics      : outputs/metrics/  (results_summary.csv, test_metrics.json, ...)"
echo "  Figures      : outputs/figures/"
echo "  Predictions  : outputs/predictions/predictions.csv"
echo "  Logs         : logs/"
