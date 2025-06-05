#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle

import numpy as np

from mpi4py import MPI

from ..rxmc import mcmc


def validate_pickle_file(path_str: str):
    """
    Validates a given path to ensure it points to a valid pickle file.

    Parameters:
        path_str (str): The file path string to validate as a pickle file.

    Returns:
        tuple:
            - A tuple containing the loaded pickle object.
            - The corresponding Path object.

    Raises:
        argparse.ArgumentTypeError:
            - If the file does not exist.
            - If the file is not a valid pickle file.
    """
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


def parse_options(comm):
    """
    Parses command line arguments for the MCMC setup.

    Parameters:
        comm (mpi4py.MPI.Comm): The MPI communicator object.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    Raises:
        ValueError:
            - If the output path is an existing file.
        argparse.ArgumentTypeError:
            - For invalid or missing methods/attributes in pickled objects.
    """
    parser = argparse.ArgumentParser(
        description="Run MCMC with independent walkers each on their own " "MPI ranks"
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
        help="Directory to write output to. If it doesn't exist it will be "
        "created. All outputs will be written to it as .np files",
    )
    parser.add_argument(
        "--write_batch_chains",
        type=bool,
        default=False,
        help="Write each batch to disk in case the program crashes",
    )
    parser.add_argument(
        "--proposal_cov_scale_factor",
        type=float,
        default=100,
        help="ratio of diagonal elements in prior covariance to corresponding"
        " diagonal elements in proposal distribution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for rank 0. All ranks will use seed + rank as their seed.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose printing",
    )
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
    """
    The main function to initiate and control the MCMC simulation using
    MPI. Gathers results and saves output.

    Raises:
        ValueError:
            - When parsing command-line options.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_options(comm)
    logl, chain, acc_frac = mcmc.run_chain(
        args.prior,
        args.corpus,
        args.nsteps,
        args.batch_size,
        args.burnin,
        args.seed,
        rank=rank,
        proposal_cov_scale_factor=args.proposal_cov_scale_factor,
        verbose=args.verbose,
        output=args.output if args.write_batch_chains else None,
    )
    # MPI Gather
    all_logl = comm.gather(logl, root=0)
    all_chains = comm.gather(chain, root=0)
    acc_frac = comm.gather(acc_frac, root=0)

    if rank == 0:
        all_chains = np.array(all_chains)
        all_logl = np.array(all_logl)
        acc_frac = np.array(acc_frac)
        print(f"\nFinished sampling all {len(all_chains)} chains.")
        print(f"Chain shape: {all_chains.shape}")
        print(f"Average acceptance fraction: {np.mean(acc_frac):.3f}")
        for i, af in enumerate(acc_frac):
            print(f"  Chain {i}: acceptance fraction = {af:.3f}")

    np.save(Path(args.output) / "all_chains.npy", all_chains)
    np.save(Path(args.output) / "log_likelihood.npy", all_logl)


if __name__ == "__main__":
    main()
