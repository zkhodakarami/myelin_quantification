#!/usr/bin/env bash
# ===========================================================================
# run_training.sh
# ===========================================================================
# End-to-end nnU-Net v2 training pipeline for white-matter segmentation on
# histology images.
#
# Prerequisites:
#   1. Python environment with nnunetv2 installed  (pip install nnunetv2)
#   2. Dataset prepared by  prepare_dataset.py
#   3. Custom trainer installed by  install_trainer.py
#   4. CUDA-capable GPU
#
# Usage:
#   bash run_training.sh            # train all 5 folds
#   bash run_training.sh 0          # train only fold 0
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Environment variables required by nnU-Net ----------------------------
export nnUNet_raw="${SCRIPT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${SCRIPT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${SCRIPT_DIR}/nnUNet_results"

# ---- Configuration --------------------------------------------------------
DATASET_ID=1
CONFIGURATION="2d"
TRAINER="nnUNetTrainerHistoAug"
FOLDS="${1:-all}"          # pass a fold number (0-4) or leave blank for all

echo "============================================================"
echo "  nnU-Net v2 Training Pipeline  --  WM Histology Seg"
echo "============================================================"
echo "  nnUNet_raw          : ${nnUNet_raw}"
echo "  nnUNet_preprocessed : ${nnUNet_preprocessed}"
echo "  nnUNet_results      : ${nnUNet_results}"
echo "  Dataset ID          : ${DATASET_ID}"
echo "  Configuration       : ${CONFIGURATION}"
echo "  Trainer             : ${TRAINER}"
echo "  Folds               : ${FOLDS}"
echo "============================================================"
echo ""

# ---- Step 1: Install custom trainer (idempotent) -------------------------
echo ">>> Installing custom trainer ..."
python "${SCRIPT_DIR}/install_trainer.py"
echo ""

# ---- Step 2: Plan & preprocess -------------------------------------------
echo ">>> Running experiment planning and preprocessing ..."
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
echo ""

# ---- Step 3: Train -------------------------------------------------------
if [ "${FOLDS}" = "all" ]; then
    for FOLD in 0 1 2 3 4; do
        echo ">>> Training fold ${FOLD} / 4 ..."
        nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} -tr ${TRAINER}
        echo ""
    done
else
    echo ">>> Training fold ${FOLDS} ..."
    nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLDS} -tr ${TRAINER}
    echo ""
fi

# ---- Step 4: Find best configuration -------------------------------------
echo ">>> Finding best configuration ..."
nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIGURATION} -tr ${TRAINER}
echo ""

echo "============================================================"
echo "  Training complete!"
echo "  Results are in: ${nnUNet_results}"
echo "============================================================"
