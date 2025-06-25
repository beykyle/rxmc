from collections import OrderedDict
from json import load, dumps
from pathlib import Path

import pandas as pd
import numpy as np


class Parameter:
    def __init__(
        self, name, dtype=float, unit="", latex_name=None, bounds=(-np.inf, np.inf)
    ):
        """
        Parameters:
            name (str): Name of the parameter
            dtype (np.dtype): Data type of the parameter
            unit (str): Unit of the parameter
            latex_name (str): LaTeX representation of the parameter
            bounds (tuple, optional): Bounds for the parameter as a tuple (min, max)
        """
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
    with open(fpath, "w") as file:
        file.write(dumps(dict(sample), indent=4))


def read_sample_from_json(fpath: Path):
    try:
        with open(fpath, "r") as file:
            return load(file, object_pairs_hook=OrderedDict)
    except IOError as exc:
        raise ValueError(f"Error: failed to open {fpath}") from exc


def array_to_list(samples: np.ndarray, params: list):
    return [to_ordered_dict(sample, params) for sample in samples]


def list_to_array(samples: list, params_dtype: tuple):
    return np.array([(sample.values()) for sample in samples], dtype=params_dtype)


def to_ordered_dict(sample: np.ndarray, params: list):
    return OrderedDict(zip(params, sample))


def list_to_dataframe(samples: list):
    return pd.DataFrame(samples)


def dataframe_to_list(samples: pd.DataFrame):
    return samples.to_dict(orient="records", into=OrderedDict)


def dump_samples_to_numpy(fpath: Path, samples: list):
    list_to_array(samples).save(fpath)


def read_samples_from_numpy(fpath: Path):
    return array_to_list(np.load(fpath))
