# The environment of every cluster task.  Sourced by job.sbatch.

# The venv: submit.py --venv exports RXMC_VENV to the job (else it is inherited
# from the submitting shell, else ~/rxmc-venv).  It must hold
# rxmc[validation,samplers] and be built from a Python that the compute nodes
# can reach at the same path; no module load is done here, since Lmod's
# `module` can trip job.sbatch's `set -u`.
RXMC_VENV="${RXMC_VENV:-$HOME/rxmc-venv}"

source "$RXMC_VENV/bin/activate"
# BLAS and torch threads to the job's cores; XLA's threads are kept on those
# cores by job.sbatch's srun --cpu-bind
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"  # torch reads OMP_NUM_THREADS too
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# a PyTensor (PyMC) compile cache per task: concurrent tasks sharing one in
# $HOME over NFS can stall on its lock
export PYTENSOR_FLAGS="compiledir=${TMPDIR:-/tmp}/pytensor-${SLURM_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"
export MPLBACKEND=Agg
