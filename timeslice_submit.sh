#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --account=neutrino:default
#SBATCH --partition=milano
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/timeslice_slurm-%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --gpus=0
#SBATCH --array=0-60
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/TimeSlicer.py /sdf/data/neutrino/summer25/ktwall/tms_timing/multihit/multihit_detector_sim_lossy_25${SLURM_ARRAY_TASK_ID}.npz /sdf/data/neutrino/summer25/ktwall/tms_timing/time_sliced/corrected/ 25${SLURM_ARRAY_TASK_ID} 500 250


