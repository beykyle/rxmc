#!/bin/bash --login

#SBATCH --job-name=ume_emcee_chuq
#SBATCH --ntasks=129
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=72:00:00           

#SBATCH --mail-type=ALL
#SBATCH --mail-user=beyerkyl@msu.edu
#SBATCH --output=%x-%j.SLURMout

# --- Environment ---
module purge
module load OpenMPI
module load Miniforge3/24.3.0-0
mamba activate pympi
export X43I_DATAPATH="/mnt/home/beyerkyl/x4db/unpack_exfor-2024/X4-2024-12-31"
python -c 'import elm; print(f"Running model version: {elm.__version__}")'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# output directory
OUTPUT_DIR=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $OUTPUT_DIR

# run the damn thing
srun -n $SLURM_NTASKS  python3 ./emcee_driver.py --input ./calibration_config_ume.pkl --output $OUTPUT_DIR --chains 256  --steps 100000
