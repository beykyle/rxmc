import numpy as np

import jitr

from .physical_model import PhysicalModel


class ElasticDifferentialXSModel(PhysicalModel):
    """
    A model that predicts the elastic differential xs for a given reaction.

    Should be used within an ElasticDifferentialXSConstraint rather than a
    regular Constraint.

    Rather than taking in an `Observation` object, this model takes in a
    `jitr.xs.elastic.DifferentialWorkspace` object, which contains all
    the relevant precomputed quantities needed to calculate the differential
    cross section using the provided interaction parameters.

    The model is initialized with two interactions: one for the central
    potential and one for the spin-orbit interaction. The `calculate_subparams`
    function is used to extract the parameters needed for each interaction.
    """

    def __init__(
        self,
        quantity: str,
        interaction_central,
        interaction_spin_orbit,
        calculate_subparams,
    ):
        """
        Initialize the ElasticDifferentialXSModel with the interactions and
        a function to calculate the subparameters.
        Parameters:
        ----------
        quantity: str
            The type of quantity to be calculated (e.g., "dXS/dA",
            "dXS/dRuth", "Ay").
        interaction_central: jitr.xs.elastic.Interaction
            The interaction for the central potential.
        interaction_spin_orbit: jitr.xs.elastic.Interaction
            The interaction for the spin-orbit potential.
        calculate_subparams: callable
            A function that takes in the model parameters and the
            DifferentialWorkspace, and returns a tuple of arguments for
            the central and spin-orbit interactions.
        """
        self.quantity = quantity
        self.interaction_central = interaction_central
        self.interaction_spin_orbit = interaction_spin_orbit
        self.calculate_subparams = calculate_subparams

        if self.quantity == "dXS/dA":
            self.extractor = extract_dXS_dA
        elif self.quantity == "dXS/dRuth":
            self.extractor = extract_dXS_dRuth
        elif self.quantity == "Ay":
            self.extractor = extract_Ay

    def xs(
        self,
        ws: jitr.xs.elastic.DifferentialWorkspace,
        *params: tuple,
    ) -> jitr.xs.elastic.ElasticXS:
        """
        Calculate the differential cross section for the given parameters.
        Parameters:
        ----------
        ws : jitr.xs.elastic.DifferentialWorkspace
            The workspace containing precomputed quantities for the reaction.
        params : tuple
            The parameters of the physical model.

        Returns:
        -------
        jitr.xs.elastic.ElasticXS
            The calculated differential cross section.
        """
        args_central, args_spin_orbit = self.calculate_subparams(*params, ws)
        return ws.xs(
            interaction_central=self.interaction_central,
            interaction_spin_orbit=self.interaction_spin_orbit,
            args_central=args_central,
            args_spin_orbit=args_spin_orbit,
        )

    def evaluate(
        self,
        ws: jitr.xs.elastic.DifferentialWorkspace,
        *params: tuple,
    ) -> np.ndarray:
        """
        Evaluate the model at the given parameters.

        Parameters:
        ----------
        ws : jitr.xs.elastic.DifferentialWorkspace
            The workspace containing precomputed quantities for the reaction.
        params : tuple
            The parameters of the physical model.

        Returns:
        -------
        np.ndarray
            The evaluated differential cross section (either dXS/dA,
            dXS/dRuth or Ay depending on self.quantity).
        """
        xs = self.xs(ws, *params)
        return self.extractor(xs, ws)

    def __call__(
        self,
        ws: jitr.xs.elastic.DifferentialWorkspace,
        *params: tuple,
    ) -> np.ndarray:
        """
        Call the model to evaluate the differential cross section.

        Parameters:
        ----------
        ws : jitr.xs.elastic.DifferentialWorkspace
            The workspace containing precomputed quantities for the reaction.
        params : tuple
            The parameters of the physical model.

        Returns:
        -------
        np.ndarray
            The evaluated differential cross section.
        """
        # overwrite PhysicalModel's __call__ method because
        # we use a DifferentialWorkspace instead of an Observation
        return self.evaluate(ws, *params)


def extract_dXS_dA(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dA in b/Sr"""
    return xs.dsdo / 1000


def extract_dXS_dRuth(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts dXS/dRuth (dimensionlesss)"""
    return xs.dsdo / ws.rutherford


def extract_Ay(
    xs: jitr.xs.elastic.ElasticXS, ws: jitr.xs.elastic.DifferentialWorkspace
) -> np.ndarray:
    """Extracts Ay (dimensionless)"""
    return xs.Ay
