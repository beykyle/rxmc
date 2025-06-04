#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle

import numpy as np
from scipy import stats

from mpi4py import MPI

from rxmc import params


def validate_pickle_file(path_str: str):
    path = Path(path_str)

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")

    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"File '{path}' is not a valid pickle file: {e}"
        )

    return obj, path


def metropolis_hastings(
    x0,
    n_steps,
    log_likelihood,
    propose,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)
    chain = np.zeros((n_steps, x0.size))
    logl_chain = np.zeros((n_steps,))
    logl = log_likelihood(x0)
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = propose(x)
        logl_new = log_likelihood(x_new)
        log_ratio = min(0, logl_new - logl)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logl = logl_new
            accepted += 1

        chain[i, ...] = x
        logl_chain[i] = logl

    return np.array(chain), np.array(logl_chain), accepted


def parse_options(comm):
    parser = argparse.ArgumentParser(
        description="Run MCMC with independent walkers each on their own MPI ranks"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=1000,
        help="Total number of MCMC steps per chain (including burn-in).",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=100,
        help="Number of steps to not log at the beginning of each chain",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of steps per batch to incrementally write to file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to write output to. If it doesn't exist it will be created. All outputs will be written to it as .np files",
    )
    parser.add_argument(
        "--proposal_cov_scale_factor",
        type=float,
        default=100,
        help="ratio of diagonal elements in prior covariance to corresponding diagonal elements in proposal distribution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for rank 0. All ranks will use seed + rank as their seed.",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose printing")
    parser.add_argument(
        "--corpus_path",
        type=lambda p: validate_pickle_file(p),
        required=True,
        help="Path to constraint corpus pickle object",
    )
    parser.add_argument(
        "--prior_path",
        type=lambda p: validate_pickle_file(p),
        required=True,
        help="Path to prior distribution pickle object",
    )

    # TODO seeding

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()

            # output dir
            if args.output is None:
                args.output = "./"
            args.output = Path(args.output)
            if args.output.is_file():
                raise ValueError(f"--output ({args.output}) cannot be a file.")
            args.output.mkdir(parents=True, exist_ok=True)

            # read in prior
            prior, path = args.prior_path
            if not hasattr(prior, "logpdf") or not callable(getattr(prior, "logpdf")):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not support logpdf "
                    "(.logpdf() method missing or not callable)."
                )
            if not hasattr(prior, "cov") or not hasattr(prior, "mean"):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not have `mean` and `cov` attributes."
                )
            args.prior = prior

            # read in corpus
            corpus, path = args.corpus_path
            if not hasattr(corpus, "logpdf") or not callable(getattr(corpus, "logpdf")):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not support logpdf "
                    "(.logpdf() method missing or not callable)."
                )
            args.corpus = corpus

    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        exit(0)
    return args


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_options(comm)

    # batching
    if args.batch_size is not None:
        rem_burn = args.burnin % args.batch_size
        n_burn_batches = args.burnin // args.batch_size
        burn_batches = n_burn_batches * [args.batch_size] + (rem_burn > 0) * [rem_burn]

        rem = (args.nsteps - args.burnin) % args.batch_size
        n_full_batches = (args.nsteps - args.burnin) // args.batch_size
        batches = n_full_batches * [args.batch_size] + (rem > 0) * [rem]
    else:
        batches = [args.nsteps - args.burnin]
        burn_batches = [args.burnin]

    # RNG
    seed = args.seed + rank
    rng = np.random.default_rng(seed)

    # likelihood
    prior = args.prior
    corpus = args.corpus

    def log_likelihood(x):
        return prior.logpdf(x) + corpus.logpdf(params.to_ordered_dict(x, corpus.params))

    # proposal distribution
    proposal_cov = prior.cov / args.proposal_cov_scale_factor
    proposal_mean = np.zeros_like(prior.mean)

    def proposal(x):
        return x + stats.multivariate_normal.rvs(
            mean=proposal_mean, cov=proposal_cov, random_state=rng
        )

    # starting location
    x0 = proposal(prior.mean)

    # run burn-in
    for i, steps_in_batch in enumerate(burn_batches):
        batch_chain, _, _ = metropolis_hastings(
            x0,
            steps_in_batch,
            log_likelihood,
            proposal,
            rng=rng,
        )
        if args.verbose:
            print(
                f"Rank: {rank}. Burn-in batch {i+1}/{len(burn_batches)} completed, {steps_in_batch} steps."
            )

    # update starter location to tail of burn-in
    x0 = batch_chain[-1]

    # run real steps
    chain = []
    logl = []
    accepted = 0

    for i, steps_in_batch in enumerate(batches):
        batch_chain, batch_logl, accepted_in_batch = metropolis_hastings(
            x0,
            steps_in_batch,
            log_likelihood,
            proposal,
            rng=rng,
        )

        # diagnostics
        accepted += accepted_in_batch
        chain.append(batch_chain)
        logl.append(batch_logl)
        x0 = batch_chain[-1]
        if args.verbose:
            print(
                f"Rank: {rank}. Batch: {i+1}/{len(batches)} completed, {steps_in_batch} steps. "
                f"Acceptance frac: {accepted_in_batch/steps_in_batch}"
            )

        # update proposal distribution?

        # update unknown covariance factor estimate (Gibbs sampling)

        # write record of batch chain to disk
        np.save(Path(args.output) / f"chain_{rank}_{i}.npy", batch_chain)

    logl = np.concatenate(logl, axis=0)
    chain = np.concatenate(chain, axis=0)

    # MPI Gather
    all_logl = comm.gather(logl, root=0)
    all_chains = comm.gather(chain, root=0)
    accepted = comm.gather(accepted, root=0)

    if rank == 0:
        all_chains = np.array(all_chains)
        all_logl = np.array(all_logl)
        acc_fracs = np.array(accepted) / (args.nsteps - args.burnin)
        print(f"\nFinished sampling all {len(all_chains)} chains.")
        print(f"Chain shape: {all_chains.shape}")
        print(f"Average acceptance fraction: {np.mean(acc_fracs):.3f}")
        for i, af in enumerate(acc_fracs):
            print(f"  Chain {i}: acceptance fraction = {af:.3f}")

    np.save(Path(args.output) / "all_chains.npy", all_chains)
    np.save(Path(args.output) / "log_likelihood.npy", all_logl)


if __name__ == "__main__":
    main()
