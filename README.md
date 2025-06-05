# rxmc
Uncertainty quantification and calibration of reaction models with Markov-Chain Monte Carlo, with flexible and composable models for the likelihood and corpus of constraints.
- Curate a corpus of experimental constraints using [`exfor_tools`](https://github.com/beykyle/exfor_tools)
- Efficiently calculate corresponding observables using [`jitR`](https://github.com/beykyle/jitr)
- Choose from a variety of likelihood models, or extend the basic [`Constraint`]() class to implement your own.

An example of this code in use is in the development of the [East Lansing Model](https://github.com/beykyle/elm)

## usage

After installation (see below), the `rxmc` module can be imported in python, and an executable `mcmc` will be installed.

See examples below for how to use both.

## installation

### for development
To modify the model, first clone and build
```bash
git clone git@github.com:beykyle/rxmc.git
cd rxmc
```

Then install an editable version locally like so:

```
pip install -ve .
```

Note that `pip` will install package dependencies listed in `requirements.txt`. It is **highly recommended** that you use an isolated virtual environment (e.g. using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda/mamba](https://mamba.readthedocs.io/en/latest/)), as this action will install all of the dependencies in `requirements.txt`, at the specific version required.


For example, with `conda`:


```
conda env create -f environment.yml
conda activate myenv
pip install -ve . --no-deps
```

or, similarily, with `mamba`:
```
mamba env create -f environment.yml
mamba activate myenv
pip install -ve . --no-deps
```

or with `venv`:

```
python -m venv .rxmc
source .rxmc/bin/activate
pip install -r requirements.txt
pip install -ve .
```

The advantages of `conda` and `mamba` is that they can install heavy binary dependencies like `openmpi` required by `mpi4py`. 

Some users may want to use their own custom environment, e.g. setup using the `module` system on a cluster. If you don't want to create an isolated environment for `rxmc`, but also don't want `pip` to overwrite the package versions you havein your environment with the ones in `requirements.txt`, you can

```
pip install -ve . --no-deps
```
This will require that your current python environment satisfies `requirements.txt`. 

## test

```
python -m unittest discover ./test
```
## examples

Check out the demos: 
- [`examples/linear_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/linear_calibration_demo.ipynb) for an illustrative example of fitting a line to data with various likelihood models.
- [`examples/parallel_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/parallel_calibration_demo.ipynb) for ways to run multiple chains in parallel, using `mpi4py` and `ipyparallel`, or using the provided `mcmc`
- [`systematic_err_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/systematic_err_demo.ipynb) for a demonstration of the likelihood models built into `rxmc`, and how to use them for situations involving systematic errors and multiple independent experimental constraints
- [`fitting_an_optical_potential.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/fitting_an_optical_potential.ipynb) for a demonstration of fitting an optical potential to multiple experimental data sets parsed from Exfor.

Other examples demonstrating actual reaction model fitting coming soon.

For a fully fledged example of this code for calibrating an uncertainty quantified global optical potential, check out [East Lansing Model](https://github.com/beykyle/elm)
