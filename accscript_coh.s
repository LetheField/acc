#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=50:00
#SBATCH --mem=16GB
#SBATCH --output=feedback_%j.out
#SBATCH --mail-user=ehgnehznait@gmail.com

module purge
module load python3/intel/3.6.3
source $HOME/tian/virtEnv/bin/activate

RUNDIR=$HOME/acc
OUTDIR=$SCRATCH/acc/coh
mkdir -p $OUTDIR

cd $RUNDIR

echo "Job starts at $(date)"
echo "SLURM_ARRAY_TASK_ID:"${SLURM_ARRAY_TASK_ID}
echo "SLURM_ARRAY_JOB_ID:"${SLURM_ARRAY_JOB_ID}

python3 acc.py --taskID $SLURM_ARRAY_TASK_ID --OUTDIR $OUTDIR --coh_c
echo "Job ends at $(date)"
