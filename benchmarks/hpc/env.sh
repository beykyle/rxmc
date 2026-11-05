# Site settings for the cluster jobs: edit these once.  Sourced by job.sbatch.

# --- edit -------------------------------------------------------------------
# module load python/3.12            # whatever provides Python 3.12 here
RXMC_VENV="${RXMC_VENV:-$HOME/rxmc-venv}"   # a venv with rxmc[validation,samplers]
# ----------------------------------------------------------------------------

source "$RXMC_VENV/bin/activate"
# one job per allocation: let XLA, BLAS and torch use the job's cores, no more
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCH_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
