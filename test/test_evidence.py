import unittest
from multiprocessing import Pool

try:
    import ipyparallel

    ipparallel_available = True
except ImportError:
    ipparallel_available = False

import numpy as np

from rxmc.physical_model import Polynomial
from rxmc.likelihood_model import LikelihoodModel
from rxmc.observation import Observation
from rxmc.constraint import Constraint
from rxmc.evidence import Evidence
from rxmc.likelihood_model import log_likelihood


class TestEvidence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Start ipyparallel cluster if available
        if ipparallel_available:
            cls.cluster = ipyparallel.Cluster(n=4)
            rc = cls.cluster.start_and_connect_sync()
            cls.client = rc
        else:
            cls.client = None

    @classmethod
    def tearDownClass(cls):
        if ipparallel_available and cls.cluster:
            cls.cluster.stop_cluster_sync()

    def setUp(self):
        y = np.array([1, 2, 3])
        x = np.array([1, 2, 3])
        y_stat_err = np.array([0.1, 0.2, 0.3])
        self.observations = [
            Observation(
                x=x,
                y=y,
                y_stat_err=y_stat_err,
            ),
        ]
        self.pm = Polynomial(order=1)
        self.constraints = [
            Constraint(
                observations=self.observations,
                physical_model=self.pm,
                likelihood_model=LikelihoodModel(),
            ),
            Constraint(
                observations=self.observations,
                physical_model=self.pm,
                likelihood_model=LikelihoodModel(),
            ),
            Constraint(
                observations=self.observations,
                physical_model=self.pm,
                likelihood_model=LikelihoodModel(),
            ),
            Constraint(
                observations=self.observations,
                physical_model=self.pm,
                likelihood_model=LikelihoodModel(),
            ),
        ]
        self.weights = np.array([1.0, 1.0, 1.0, 1.0])

        self.evidence = Evidence(
            constraints=self.constraints,
            weights=self.weights,
        )
        self.model_params = (3, 5)
        modely = self.pm.evaluate(self.observations[0], *self.model_params)
        delta = y - modely
        chi2 = np.sum((delta / y_stat_err) ** 2)
        N = self.observations[0].n_data_pts
        log_det = np.log(np.linalg.det(self.observations[0].statistical_covariance))
        logl_single_constraint = -0.5 * (N * np.log(2 * np.pi) + log_det + chi2)
        self.expected_loglikelihood = 4 * logl_single_constraint

    def test_serial_execution(self):
        log_likelihood = self.evidence.log_likelihood(model_params=self.model_params)
        self.assertEqual(log_likelihood, self.expected_loglikelihood)

    def test_multiprocessing_execution(self):
        with Pool(processes=2) as pool:
            log_likelihood = self.evidence.log_likelihood(
                model_params=self.model_params,
                executor=pool,
            )
        self.assertEqual(log_likelihood, self.expected_loglikelihood)

    @unittest.skipUnless(ipparallel_available, "ipyparallel not available")
    def test_ipyparallel_execution(self):
        if self.client is not None:
            log_likelihood = self.evidence.log_likelihood(
                model_params=self.model_params,
                executor=self.client,
            )
            self.assertEqual(log_likelihood, self.expected_loglikelihood)

    def test_invalid_executor(self):
        with self.assertRaises(ValueError):
            self.evidence.log_likelihood(
                model_params=self.model_params,
                executor="invalid_executor",
            )


if __name__ == "__main__":
    unittest.main()
