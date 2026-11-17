#!/bin/bash
#SBATCH -J rn50-static-12
#SBATCH -o rn50-static-age12.log%j
#SBATCH -e rn50-static-age12.err%j
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu
#SBATCH -t 08:00:00

set -e

echo "Date       = $(date)"
echo "Hostname   = $(hostname -s)"
echo "Working Dir= $(pwd)"
echo ""
echo "Num Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated = $SLURM_NTASKS"
echo "GPU Allocated       = $SLURM_JOB_GPUS"
echo ""

source ~/.bashrc
mamba activate cap
export PYTHONPATH=$PWD:$PYTHONPATH
which python
python -V
python -c "import torch; print(torch.__file__)"
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi

stdoutFile="rn50-static-age12.log$SLURM_JOB_ID"
stderrFile="rn50-static-age12.err$SLURM_JOB_ID"
resultsFile="results_$SLURM_JOB_ID.txt"

MODEL=resnet50
AGE=12
TRAIN_CSV=train_cn12.csv
VALIDATION_CSV=val_cn12.csv
TEST_CSV=synthetic_csv/test_all.csv
CHECKPOINT_DIR=best_models_bin/move_shuffle_static
OUTPUT_DIR=eval_outputs_binary/move_shuffle_static

for grid in 1 2 4 8 16 32 64 128; do
    echo
    echo "Training $MODEL static-seed model for age cutoff $AGE, grid $grid..."
    python predictor/train_binary.py \
        -g "$grid" \
        -t "$TRAIN_CSV" \
        -v "$VALIDATION_CSV" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --model "$MODEL" \
        --no-increment-shuffle-seed

    checkpoint="$CHECKPOINT_DIR/rn50_train_cn12_g${grid}.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: Expected checkpoint not found: $checkpoint" >&2
        exit 1
    fi

    echo "Testing $MODEL for age cutoff $AGE, grid $grid..."
    python predictor/test_binary.py \
        -g "$grid" \
        --age "$AGE" \
        --model "$MODEL" \
        --checkpoint "$checkpoint" \
        --test_csv "$TEST_CSV" \
        --output_dir "$OUTPUT_DIR"
done

awk 1 "$stdoutFile" "$stderrFile" > "$resultsFile"
rm -f "$stdoutFile" "$stderrFile"
