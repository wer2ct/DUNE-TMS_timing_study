#!/bin/bash
#SBATCH --job-name=long-Bertha-delaunay
#SBATCH --nodes=1
#SBATCH --account=neutrino:ml-tms
#SBATCH --partition=turing
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/long-Bertha.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  --nv \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/TMS_Net/ModelAndTrain.py /sdf/data/neutrino/summer25/ktwall/processed/ /sdf/data/neutrino/summer25/ktwall/logs_Bertha/
