"""
Parameter definitions and sample serialization utilities.

The :class:`Parameter` class describes a single scalar model parameter —
its name, data type, physical unit, LaTeX label, and optional bounds.

The module also provides functions for converting between the various
container formats used throughout the package (``OrderedDict``, ``list``,
``np.ndarray``, ``pd.DataFrame``) and for persisting samples to JSON or
NumPy binary files.
"""

from collections import OrderedDict
from json import dumps, load
from pathlib import Path

import numpy as np
import pandas as pd


class Parameter:
    """A single scalar model parameter.

    Parameters
    ----------
    name : str
        Human-readable name of the parameter.
    dtype : type, optional
        Data type of the parameter value.  Defaults to ``float``.
    unit : str, optional
        Physical unit string (e.g. ``"MeV"``).  Defaults to ``""``.
    latex_name : str, optional
        LaTeX representation used in plots and documentation.  Defaults to
        ``name`` when not supplied.
    bounds : tuple of float, optional
        ``(lower, upper)`` bounds for the parameter.  Defaults to
        ``(-np.inf, np.inf)``.
    """

    def __init__(
        self, name, dtype=float, unit="", latex_name=None, bounds=(-np.inf, np.inf)
    ):
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.bounds = bounds
        self.latex_name = latex_name if latex_name else name

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return False
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.unit == other.unit
            and self.latex_name == other.latex_name
            and self.bounds == other.bounds
        )


def dump_sample_to_json(fpath: Path, sample: OrderedDict):
    """Write a single parameter sample to a JSON file.

    Parameters
    ----------
    fpath : Path
        Destination file path.
    sample : OrderedDict
        Mapping of parameter name to value.
    """
    with open(fpath, "w") as file:
        file.write(dumps(dict(sample), indent=4))


def read_sample_from_json(fpath: Path) -> OrderedDict:
    """Read a single parameter sample from a JSON file.

    Parameters
    ----------
    fpath : Path
        Source file path.

    Returns
    -------
    OrderedDict
        Mapping of parameter name to value, preserving insertion order.

    Raises
    ------
    ValueError
        If the file cannot be opened.
    """
    try:
        with open(fpath, "r") as file:
            return load(file, object_pairs_hook=OrderedDict)
    except IOError as exc:
        raise ValueError(f"Error: failed to open {fpath}") from exc


def array_to_list(samples: np.ndarray, params: list) -> list:
    """Convert a 2-D sample array to a list of ``OrderedDict`` objects.

    Parameters
    ----------
    samples : np.ndarray, shape (n, ndim)
        Array of parameter vectors, one row per sample.
    params : list of Parameter
        Parameter objects whose names become the dict keys.

    Returns
    -------
    list of OrderedDict
        One ordered mapping per row of *samples*.
    """
    return [to_ordered_dict(sample, params) for sample in samples]


def list_to_array(samples: list, params_dtype: tuple) -> np.ndarray:
    """Convert a list of ``OrderedDict`` samples to a structured NumPy array.

    Parameters
    ----------
    samples : list of OrderedDict
        Each element maps parameter name to value.
    params_dtype : tuple
        NumPy structured dtype describing the parameter names and types.

    Returns
    -------
    np.ndarray
        Structured array with one row per sample.
    """
    return np.array([(sample.values()) for sample in samples], dtype=params_dtype)


def to_ordered_dict(sample: np.ndarray, params: list) -> OrderedDict:
    """Zip a 1-D parameter vector with a list of :class:`Parameter` objects.

    Parameters
    ----------
    sample : np.ndarray, shape (ndim,)
        Parameter values.
    params : list of Parameter
        Parameter objects whose names become the dict keys.

    Returns
    -------
    OrderedDict
        Mapping of parameter name to value.
    """
    return OrderedDict(zip(params, sample))


def list_to_dataframe(samples: list) -> pd.DataFrame:
    """Convert a list of ``OrderedDict`` samples to a ``pandas.DataFrame``.

    Parameters
    ----------
    samples : list of OrderedDict
        Each element maps parameter name to value.

    Returns
    -------
    pd.DataFrame
        One row per sample, one column per parameter.
    """
    return pd.DataFrame(samples)


def dataframe_to_list(samples: pd.DataFrame) -> list:
    """Convert a ``pandas.DataFrame`` of samples to a list of ``OrderedDict``.

    Parameters
    ----------
    samples : pd.DataFrame
        One row per sample, one column per parameter.

    Returns
    -------
    list of OrderedDict
        One mapping per row.
    """
    return samples.to_dict(orient="records", into=OrderedDict)


def dump_samples_to_numpy(fpath: Path, samples: list):
    """Serialize a list of samples to a NumPy binary file.

    Parameters
    ----------
    fpath : Path
        Destination ``.npy`` file path.
    samples : list of OrderedDict
        Samples to persist.
    """
    list_to_array(samples).save(fpath)


def read_samples_from_numpy(fpath: Path) -> list:
    """Load samples from a NumPy binary file.

    Parameters
    ----------
    fpath : Path
        Source ``.npy`` file path.

    Returns
    -------
    list of OrderedDict
        Loaded samples.
    """
    return array_to_list(np.load(fpath))
