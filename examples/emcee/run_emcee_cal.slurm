#!/bin/bash --login

#SBATCH --job-name=rxmc_with_emcee_example
#SBATCH --ntasks=129
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=3:00:00           

# --- Environment ---
# set up python environment
pip install -r ./requirements.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# output directory
OUTPUT_DIR=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $OUTPUT_DIR

# run
srun -n $SLURM_NTASKS /usr/bin/time python3 ./emcee_driver.py --input ./rxmc_poly_demo_conf.pkl --output $OUTPUT_DIR --chains 256  --steps 20000 --batch-size 500 --step-size 2.0
