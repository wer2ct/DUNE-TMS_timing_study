#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --account=neutrino:default
#SBATCH --partition=milano
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/hitwise_slurm-%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --gpus=0
#SBATCH --array=0-9
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/HitWiseEffects.py /sdf/data/neutrino/summer25/tanaka/nd-production/run-spill-build/MicroProdN4p1_NDComplex_FHC.spill.full/EDEPSIM_SPILLS/0002000/0002500/MicroProdN4p1_NDComplex_FHC.spill.full.00025${SLURM_ARRAY_TASK_ID}* /sdf/data/neutrino/summer25/ktwall/tms_timing/hitwise/ 25${SLURM_ARRAY_TASK_ID}


