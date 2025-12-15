#!/bin/bash --login

#SBATCH --job-name=rxmc_with_emcee_example
#SBATCH --ntasks=65
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=3:00:00           

# install rxmc into python environment
# load any modules you might need here (e.g., python, gcc, etc.)
python -m pip install -r requirements.txt
python -m pip install ../../.
# NOTE: depending on your cluster setup, you might want to create 
# and activate a virtual environment instead, or use pip install --user

# set number of threads so numpy/scipy do not oversubscribe
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# output directory
OUTPUT_DIR=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $OUTPUT_DIR

srun -n $SLURM_NTASKS /usr/bin/time python3 ./emcee_driver.py \
    --input ./rxmc_poly_demo_conf.pkl \
    --output $OUTPUT_DIR \
    --chains 128  \
    --steps 2000 \
    --batch-size 100 \
    --step-size 2.0
