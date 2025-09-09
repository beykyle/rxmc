#!/bin/bash --login

#SBATCH --job-name=emcee_cal_mpi
#SBATCH --ntasks=129
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=1:00:00           

# --- Environment ---
module purge
module load OpenMPI
module load Miniforge3/24.3.0-0
mamba activate pympi
export X43I_DATAPATH="/mnt/home/beyerkyl/x4db/unpack_exfor-2024/X4-2024-12-31"
python -c 'import elm; print(f"Running model version: {elm.__version__}")'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun python3 ./run_calibration.py --input ./calibration_config.pkl --chains 128  --steps 1000 --output ./
