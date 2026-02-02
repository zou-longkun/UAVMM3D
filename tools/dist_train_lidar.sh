#!/bin/bash
# Multi-modal fusion model training script
# Purpose: Solidify training commands for one-click launch of distributed training (with mixed precision)

# 1. Set environment variables (control CPU threads, specify GPUs)
export OMP_NUM_THREADS=2  # OpenMP threads per process (adjust as needed)
export CUDA_VISIBLE_DEVICES=3,4,5  # Specify GPUs to use (GPU IDs: 3, 4, 5)

# 2. Define core training parameters (easy to modify later, match GPU count/port)
MASTER_PORT=29501  # Distributed communication port (avoid conflict with 29500)
NUM_PROC=3  # Number of processes per machine (match GPU count: 3 GPUs → 3 processes)
LOG_FILE=./train_logs/$(date +%Y%m%d_%H%M%S)_fusion.log  # Log file with timestamp (avoid overwriting)
USE_AMP="--use_amp"  # Mixed precision training parameter (enabled by default)
SEPARATOR=$(printf '=%.0s' {1..60})  # Generate 60 equal signs (compatible with all Shells)

# 3. Create log directory (auto-create if not exists, avoid log write failure)
mkdir -p ./train_logs

# 4. Execute distributed training command (FIXED: correct line break with \)
nohup torchrun \
--nproc_per_node=$NUM_PROC \
--master_port=$MASTER_PORT \
train_laam6d_fusion.py \
--launcher pytorch \
$USE_AMP \
> $LOG_FILE 2>&1 &

# 5. Print launch information
echo "$SEPARATOR"
echo "Multi-modal fusion training started successfully!"
echo "$SEPARATOR"
echo "Current Configuration:"
echo "  GPU IDs: ${CUDA_VISIBLE_DEVICES}"
echo "  Number of Processes: ${NUM_PROC} (matches GPU count)"
echo "  OpenMP Threads per Process: ${OMP_NUM_THREADS}"
echo "  Communication Port: ${MASTER_PORT}"
echo "  Mixed Precision: Enabled (Parameter: ${USE_AMP})"
echo "$SEPARATOR"
echo "Log File Path: ${LOG_FILE}"
echo "View Real-time Logs: tail -f ${LOG_FILE}"
echo "View Training Processes: ps aux | grep train_laam6d_fusion.py"
echo "Stop Training: pkill -f train_laam6d_fusion.py (use with caution)"
echo "\nStarting real-time log monitoring..."
tail -f ${LOG_FILE}
echo "$SEPARATOR"