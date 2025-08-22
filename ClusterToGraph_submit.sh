#!/bin/bash
#SBATCH --job-name=FCgraphupdate2400
#SBATCH --nodes=1
#SBATCH --account=neutrino:default
#SBATCH --partition=milano
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/FCclustertograph_slurm-%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:00:00
#SBATCH --gpus=0
#SBATCH --array=0-99
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/TMS_Net/ClusterToGraph.py /sdf/data/neutrino/summer25/tanaka/nd-production/run-spill-build/MicroProdN4p1_NDComplex_FHC.spill.full/EDEPSIM_SPILLS/0002000/0002400/MicroProdN4p1_NDComplex_FHC.spill.full.00024${SLURM_ARRAY_TASK_ID}* /sdf/data/neutrino/summer25/ktwall/tms_timing/multihit/multihit_detector_sim_lossy_24${SLURM_ARRAY_TASK_ID}* /sdf/data/neutrino/summer25/ktwall/tms_timing/dbscan_clustered/hits_DBSCAN_clustered_epsilon_0.1_24${SLURM_ARRAY_TASK_ID}* 24${SLURM_ARRAY_TASK_ID}
