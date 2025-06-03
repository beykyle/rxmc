from collections import OrderedDict
from json import load, dumps
from pathlib import Path

import pandas as pd
import numpy as np


class Parameter:
    def __init__(self, name, dtype, unit, latex_name):
        """
        Parameters:
            name (str): Name of the parameter
            dtype (np.dtype): Data type of the parameter
            unit (str): Unit of the parameter
            latex_name (str): LaTeX representation of the parameter
        """
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.latex_name = latex_name


def dump_sample_to_json(fpath: Path, sample: OrderedDict):
    with open(fpath, "w") as file:
        file.write(dumps(dict(sample), indent=4))


def read_sample_from_json(fpath: Path):
    try:
        with open(fpath, "r") as file:
            return load(file, object_pairs_hook=OrderedDict)
    except IOError:
        raise f"Error: failed to open {fpath}"


def array_to_list(samples: np.ndarray, params: list):
    return [to_ordered_dict(sample, params) for sample in samples]


def list_to_array(samples: list, params_dtype: tuple):
    return np.array([(sample.values()) for sample in samples], dtype=params_dtype)


def to_ordered_dict(sample: np.ndarray, params: list):
    return OrderedDict(zip([p.name for p in params], sample))


def list_to_dataframe(samples: list):
    return pd.DataFrame(samples)


def dataframe_to_list(samples: pd.DataFrame):
    return samples.to_dict(orient="records", into=OrderedDict)


def dump_samples_to_numpy(fpath: Path, samples: list):
    list_to_array(samples).save(fpath)


def read_samples_from_numpy(fpath: Path):
    return array_to_list(np.load(fpath))
