#!/bin/bash
#SBATCH	-J job16 		# job name
#SBATCH -o data-job16.log%j	# output and error file name (%j expands to jobID)
#SBATCH -c 2			# cpus per task
#SBATCH -N 1			# num nodes to run on
#SBATCH --gres=gpu:L40:1	# request a gpu
#SBATCH -p gpu			# queue	(partition)
#SBATCH -t 06:00:00		# run time (hh:mm:ss)

echo "Date		= $(date)"
echo "Hostname		= $(hostname -s)"
echo "Working Dir	= $(pwd)"
echo ""
echo "Num Nodes Allocated 	= $SLURM_JOB_NUM_NODES"
echo "Num Tasks Allocated	= $SLURM_NTASKS"
echo "GPU Allocated		= $SLURM_JOB_GPUS"
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

stdoutFile="data_job1.o$SLURM_JOB_ID"
stderrFile="data_job1.e$SLURM_JOB_ID"
logFile="data-job1.log$SLURM_JOB_ID"
resultsFile="results_$SLURM_JOB_ID.txt"

# Run Program
python ./dataset/generation/diffusion_gen.py -s 66 -e 99 -n 500 -c 7140000 --tags-file tags_job16.csv

awk 1 "$logFile" "$stdoutFile" "$stderrFile" > "$resultsFile"
rm -f "$logFile" "$stdoutFile" "$stderrFile"
