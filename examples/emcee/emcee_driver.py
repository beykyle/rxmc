import argparse
import datetime
import sys
from pathlib import Path
from time import time

import emcee
import mpi4py.MPI as MPI
import numpy as np
from schwimmbad import MPIPool

import posterior


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dill-pickled CalibrationConfig",
    )
    parser.add_argument("--chains", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=str, default="./")
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["joint"], default="joint")
    parser.add_argument("--serial-timing-test", action="store_true")
    parser.add_argument("--MPI-timing-test", action="store_true")
    parser.add_argument("--step-size", type=float, default=2.0)
    return parser.parse_args()


def run_joint(args, pool, size=1):
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    backend_path = output_path / "chains.h5"

    # Check if the backend file already exists and handle it appropriately.
    if backend_path.exists():
        # File exists: load the backend to continue the sampling process.
        backend = emcee.backends.HDFBackend(backend_path)

        if backend.iteration == 0:
            # If, for some reason, the backend exists but has zero iterations.
            print("Backend found but has zero iterations; resetting...")
            backend.reset(args.chains, ndim=posterior.NDIM)
            p0 = posterior.starting_location(args.chains)
        else:
            # Normal case: Backend exists and has prior progress.
            print(f"Resuming from iteration {backend.iteration}")
            p0 = backend.get_last_sample().coords
    else:
        # File does not exist: create a new backend and reset it.
        backend = emcee.backends.HDFBackend(backend_path)
        print("Initializing new backend...")
        backend.reset(args.chains, ndim=posterior.NDIM)
        p0 = posterior.starting_location(args.chains)

    sampler = emcee.EnsembleSampler(
        args.chains,
        posterior.NDIM,
        posterior.log_posterior,
        pool=pool,
        backend=backend,
        moves=emcee.moves.StretchMove(a=args.step_size),
    )

    # burn-in
    if backend.iteration < args.burnin:
        print(f"Running burn-in for {args.burnin - backend.iteration} steps...")
        state = sampler.run_mcmc(p0, args.burnin - backend.iteration, progress=False)
        sampler.reset()
        p0 = state
        print("Burn-in complete.")

    # production run
    chunk = args.batch_size or args.steps
    index = 0
    autocorr = np.empty(args.steps)
    old_tau = np.inf
    print(
        f"Starting production sampling for {args.steps} steps on {args.chains} chains "
        f"on {size - 1} MPI ranks.\n"
    )
    sys.stdout.flush()

    t0 = time()
    for _ in sampler.sample(p0, iterations=args.steps):
        if sampler.iteration % chunk:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print(f"Chains converged after {sampler.iteration} steps")
            break
        else:
            print(
                f"Step {sampler.iteration}: mean autocorr time = {np.mean(tau):.1f} "
                f"steps"
            )
        sys.stdout.flush()
        old_tau = tau
        dt = time() - t0

    print(f"Sampling took {datetime.timedelta(seconds=dt)}")


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    args = parse_args()

    posterior.init_posterior(args.input)

    # warm up JIT on every rank
    p0 = posterior.starting_location(args.chains)
    posterior.log_posterior(p0[0])

    # timing test serial
    if args.serial_timing_test:
        t0 = time()
        posterior.log_posterior_batch(p0)
        dt = time() - t0
        print(f"{args.chains} samples in {datetime.timedelta(seconds=dt)} [hh:mm:ss]")

    # timing test parallel
    if args.MPI_timing_test:
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            else:
                chunk_size = args.chains // (size - 1)
                inputs = [
                    posterior.starting_location(chunk_size) for _ in range(size - 1)
                ]
                print(
                    f"Timing {args.chains} samples on {size - 1} ranks with chunk size {chunk_size}..."
                )

                t0 = time()
                pool.map(posterior.log_posterior_batch, inputs)
                dt = time() - t0
                print(
                    f"{args.chains} samples on {size - 1} ranks in {datetime.timedelta(seconds=dt)} [hh:mm:ss]"
                )

    if not args.serial_timing_test and not args.MPI_timing_test:
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            else:
                if args.mode == "joint":
                    run_joint(args, pool, size=size)
                else:
                    raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
