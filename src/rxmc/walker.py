"""
Walker: end-to-end Gibbs-style MCMC for Bayesian calibration.

:class:`Walker` orchestrates one :class:`~rxmc.param_sampling.Sampler` for the
physical-model parameters and optionally several samplers for parametric
likelihood parameters, alternating between them in a Gibbs framework.  It is
intended for smaller-scale prototyping and validation problems; for large
production calibrations prefer :class:`~rxmc.config.CalibrationConfig` with
an external sampler.
"""

import numpy as np

from .evidence import Evidence
from .param_sampling import Sampler


class Walker:
    """Gibbs-style MCMC coordinator for a Bayesian calibration problem.

    Manages one sampler for the physical-model parameters and, optionally,
    per-constraint samplers for parametric likelihood parameters.  The samplers
    alternate in a Gibbs framework: model parameters are updated with the
    likelihood parameters held fixed, then each set of likelihood parameters is
    updated with the model parameters held fixed.

    Parameters
    ----------
    model_sampler : Sampler
        Sampler for the physical-model parameters.
    evidence : Evidence
        Evidence object containing the observations and likelihood models.
    likelihood_samplers : list of Sampler, optional
        One sampler per entry in ``evidence.parametric_constraints``.
    rng : np.random.Generator, optional
        Random number generator.  Defaults to ``default_rng(42)``.

    Raises
    ------
    ValueError
        If the physical-model parameters in *evidence* and *model_sampler* do
        not match.
    ValueError
        If the number of *likelihood_samplers* does not equal the number of
        parametric constraints in *evidence*.
    ValueError
        If any likelihood sampler's parameters do not match those of the
        corresponding parametric constraint.
    """

    def __init__(
        self,
        model_sampler: Sampler,
        evidence: Evidence,
        likelihood_samplers: list[Sampler] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.model_sampler = model_sampler
        self.likelihood_samplers = likelihood_samplers or []
        self.evidence = evidence
        self.rng = rng if rng is not None else np.random.default_rng(42)

        self.gibbs_sampling = len(self.likelihood_samplers) > 0

        if self.evidence.model_params != self.model_sampler.params:
            raise ValueError(
                "Inconsistent physical model parameters between "
                "'evidence' and 'model_sampler'"
            )

        if len(self.likelihood_samplers) != len(self.evidence.parametric_constraints):
            raise ValueError(
                "The lists 'likelihood_samplers' and "
                "'evidence.parametric_constraints' must correspond!"
            )
        for i, conf in enumerate(self.likelihood_samplers):
            constraint = self.evidence.parametric_constraints[i]
            if list(constraint.params) != list(conf.params):
                raise ValueError(
                    "Inconsistent likelihood model parameters "
                    f"between 'likelihood_samplers[{i}]' and "
                    f"'evidence.parametric_constraints[{i}]'"
                )

    def run_model_batch(self, n_steps, x0, likelihood_params=None, burn=False):
        """Sample model parameters for fixed likelihood parameters.

        Parameters
        ----------
        n_steps : int
            Number of MCMC steps.
        x0 : np.ndarray
            Starting location for the model parameters.
        likelihood_params : list of tuple, optional
            Fixed values of the likelihood parameters for each parametric
            constraint.  Defaults to ``[]``.
        burn : bool, optional
            If ``True``, treat as burn-in (samples are not recorded).
        """
        likelihood_params = likelihood_params or []
        self.model_sampler.sample(
            n_steps,
            x0,
            self.rng,
            lambda x: self.log_posterior(x, likelihood_params),
            burn=burn,
        )

    def run_likelihood_batches(
        self, n_steps, starting_locations, model_params, burn=False
    ):
        """Sample each set of likelihood parameters for fixed model parameters.

        Parameters
        ----------
        n_steps : int
            Number of MCMC steps per likelihood sampler.
        starting_locations : list of np.ndarray
            Starting locations for each likelihood sampler.
        model_params : tuple
            Fixed physical-model parameter values.
        burn : bool, optional
            If ``True``, treat as burn-in (samples are not recorded).
        """
        wmll = self.evidence.weighted_marginal_log_likelihood
        for i, sampler in enumerate(self.likelihood_samplers):
            constraint = self.evidence.parametric_constraints[i]
            ym = constraint.predict(*model_params)

            def log_posterior_lm(x, sampler=sampler, i=i, ym=ym):
                lp = sampler.prior.logpdf(x) + wmll(i, ym, *np.atleast_1d(x))
                return float(np.squeeze(lp))

            x0 = starting_locations[i]
            sampler.sample(n_steps, x0, self.rng, log_posterior_lm, burn=burn)

    def log_likelihood(self, model_params, likelihood_params):
        return self.evidence.log_likelihood(model_params, likelihood_params)

    def log_posterior(self, model_params, likelihood_params):
        return self.log_likelihood(model_params, likelihood_params) + self.log_prior(
            model_params, likelihood_params
        )

    def log_prior(self, model_params, likelihood_params):
        """Log prior probability of model and likelihood parameters.

        Parameters
        ----------
        model_params : tuple
            Physical-model parameter values.
        likelihood_params : list of tuple
            One tuple of likelihood parameter values per parametric constraint.

        Returns
        -------
        float
            Sum of log prior densities for model and likelihood parameters.
        """
        lp = self.model_sampler.prior.logpdf(model_params)
        lp += sum(
            lm.prior.logpdf(likelihood_params[i])
            for i, lm in enumerate(self.likelihood_samplers)
        )
        return float(np.squeeze(lp))

    def walk(
        self,
        n_steps: int,
        burnin: int = 0,
        batch_size: int = None,
        verbose: bool = True,
    ):
        """Run the full MCMC chain.

        Updates the internal state of ``model_sampler`` and each entry of
        ``likelihood_samplers`` with the accumulated chain, log posteriors,
        and acceptance statistics.

        Parameters
        ----------
        n_steps : int
            Total number of active (post-burn-in) steps.
        burnin : int, optional
            Number of burn-in steps discarded before recording.
            Defaults to ``0``.
        batch_size : int, optional
            Steps per batch.  If ``None`` the entire chain is one batch.
        verbose : bool, optional
            Print batch completion messages.  Defaults to ``True``.
        """
        if batch_size is not None:
            rem_burn = burnin % batch_size
            n_burn_batches = burnin // batch_size
            burn_batches = n_burn_batches * [batch_size] + (rem_burn > 0) * [rem_burn]

            rem = n_steps % batch_size
            n_full_batches = n_steps // batch_size
            batches = n_full_batches * [batch_size] + (rem > 0) * [rem]
        else:
            batches = [n_steps]
            burn_batches = [burnin]

        if burnin == 0:
            burn_batches = []

        for i, steps_in_batch in enumerate(burn_batches):
            self.run_model_batch(
                steps_in_batch,
                self.model_sampler.state,
                [sampler.state for sampler in self.likelihood_samplers],
                burn=True,
            )

            if self.gibbs_sampling:
                self.run_likelihood_batches(
                    steps_in_batch,
                    [sampler.state for sampler in self.likelihood_samplers],
                    self.model_sampler.state,
                    burn=True,
                )

            if verbose:
                print(
                    f"Burn-in batch {i + 1}/{len(burn_batches)}"
                    f" completed, {steps_in_batch} steps."
                )

        for i, steps_in_batch in enumerate(batches):
            self.run_model_batch(
                steps_in_batch,
                self.model_sampler.state,
                [sampler.state for sampler in self.likelihood_samplers],
            )

            if self.gibbs_sampling:
                self.run_likelihood_batches(
                    steps_in_batch,
                    [sampler.state for sampler in self.likelihood_samplers],
                    self.model_sampler.state,
                )

            if verbose:
                msg = (
                    f"Batch: {i + 1}/{len(batches)} completed, "
                    f"{steps_in_batch} steps. "
                    f"\n  Model parameter acceptance fraction: "
                    f"{self.model_sampler.most_recent_batch_acceptance_fraction():.3f}"
                )
                if self.gibbs_sampling:
                    msg += (
                        f"\n  Likelihood parameter acceptance fractions: "
                        f"{[sampler.most_recent_batch_acceptance_fraction() for sampler in self.likelihood_samplers]}"
                    )
                print(msg)
