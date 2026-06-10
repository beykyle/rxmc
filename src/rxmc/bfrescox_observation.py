"""
Observation class for Bfrescox

"""

from pathlib import Path

import bfrescox
import numpy as np
from pandas import DataFrame

from .observation import Observation


class BfrescoxObservation(Observation):
    """
    Observation that represents a single output from a Bfrescox calculation,
    such as an elastic differential cross section
    (dXS/dΩ, dXS/dRuth, or analysing power Ay) at energy and set of angles.

    TODO: allow for multiple observations from a single bfrescox run
    (e.g. Ay and dXS/dΩ at the same energy and angles)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        template_path: Path | str,
        runtime_path: Path | str | None = None,
        y_stat_err=None,
        y_sys_err_normalization=None,
        y_sys_err_normalization_mask=None,
        y_sys_err_offset=None,
        y_sys_err_offset_mask=None,
    ):
        self.template_path = Path(template_path)
        self.runtime_path = (
            Path(runtime_path) if runtime_path is not None else Path("./")
        )

        super().__init__(
            x,
            y,
            y_stat_err=y_stat_err,
            y_sys_err_normalization=y_sys_err_normalization,
            y_sys_err_normalization_mask=y_sys_err_normalization_mask,
            y_sys_err_offset=y_sys_err_offset,
            y_sys_err_offset_mask=y_sys_err_offset_mask,
        )

    def run(self, params_dict: dict) -> dict[str, DataFrame]:
        """
        Run the Bfrescox calculation with the given parameters
        and return the predicted observable as a DataFrame.
        """
        cfg = bfrescox.Configuration.from_template(
            self.template_path,
            self.runtime_path.joinpath("frescox.in"),
            params_dict,
            overwrite=True,
        )
        bfrescox.run_simulation(
            cfg,
            self.runtime_path.joinpath("frescox.out"),
            cwd=self.runtime_path,
            overwrite=True,
        )
        results = bfrescox.parse_fort16(self.runtime_path.joinpath("fort.16"))
        return results
