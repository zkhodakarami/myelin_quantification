#!/usr/bin/env bash
# ===========================================================================
# run_inference.sh
# ===========================================================================
# Run nnU-Net v2 inference on the held-out test set and then evaluate the
# predictions against ground-truth masks.
#
# Usage:
#   bash run_inference.sh                      # default test set
#   bash run_inference.sh /path/to/input_dir   # custom input directory
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Environment variables ------------------------------------------------
export nnUNet_raw="${SCRIPT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${SCRIPT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${SCRIPT_DIR}/nnUNet_results"

# ---- Configuration --------------------------------------------------------
DATASET_ID=1
CONFIGURATION="2d"
TRAINER="nnUNetTrainerHistoAug"
PLANS="nnUNetPlans"

INPUT_DIR="${1:-${nnUNet_raw}/Dataset001_WMHistoSeg/imagesTs}"
OUTPUT_DIR="${SCRIPT_DIR}/predictions"
GT_DIR="${SCRIPT_DIR}/test_ground_truth"

echo "============================================================"
echo "  nnU-Net v2 Inference  --  WM Histology Seg"
echo "============================================================"
echo "  Input  : ${INPUT_DIR}"
echo "  Output : ${OUTPUT_DIR}"
echo "  GT dir : ${GT_DIR}"
echo "============================================================"
echo ""

# ---- Step 1: Predict -----------------------------------------------------
echo ">>> Running prediction (ensemble of all 5 folds) ..."
nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d ${DATASET_ID} \
    -c ${CONFIGURATION} \
    -tr ${TRAINER} \
    -p ${PLANS} \
    --save_probabilities
echo ""

# ---- Step 2: Evaluate ----------------------------------------------------
if [ -d "${GT_DIR}" ]; then
    echo ">>> Evaluating predictions against ground truth ..."
    python "${SCRIPT_DIR}/evaluate.py" \
        --pred_dir "${OUTPUT_DIR}" \
        --gt_dir "${GT_DIR}" \
        --output_csv "${SCRIPT_DIR}/evaluation_results.csv" \
        --vis_dir "${SCRIPT_DIR}/visualisations"
    echo ""
else
    echo "Ground-truth directory not found (${GT_DIR}). Skipping evaluation."
fi

echo "============================================================"
echo "  Inference complete!"
echo "  Predictions : ${OUTPUT_DIR}"
echo "============================================================"
