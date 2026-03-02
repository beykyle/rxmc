[![Python package](https://github.com/beykyle/rxmc/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/rxmc/actions/workflows/python-package.yml)


# RxMC
Bayesian calibration of reaction models, with flexible and composable models for the likelihood and body of evidence.
- curate the corpus of experimental constraints (e.g. using [`exfor_tools`](https://github.com/beykyle/exfor_tools))
- efficiently calculate your model's corresponding predictions for these observables using [`jitR`](https://github.com/beykyle/jitr)
- choose from a variety of likelihood models, or extend the basic [`LikelihoodModel`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/likelihood_model.py) class to implement your own.
- package the constraints, physics model, solver, and likelihood model together in a [`Evidence`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/evidence.py) object which provides the likelihood of a given model parameter, for use in Bayesian calibration 
- run Bayesian calibration using a [`Walker`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/walker.py), or using a 3rd party MCMC sampler like [emcee](https://emcee.readthedocs.io/en/stable/user/sampler/).

Check out the [`examples/` directory](https://github.com/beykyle/rxmc/blob/main/examples/).


## installation
### pypi

```
pip install rxmc
```


### for development
```bash
git clone git@github.com:beykyle/rxmc.git
cd rxmc
```

Then install an editable version locally like so:

```
pip install -ve .
```

It is **highly recommended** that you use an isolated virtual environment (e.g. using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda/mamba](https://mamba.readthedocs.io/en/latest/)), as this action will install all of the dependencies in `requirements.txt`, at the specific version required.


For example, with `conda`:


```
conda env create -f environment.yml
conda activate rxmc
pip install -ve . --no-deps
```

or, similarily, with `mamba`:
```
mamba env create -f environment.yml
mamba activate rxmc
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
pytest
```

The examples and demos also serve as tests:

```
pip install -r examples/requirements.txt
pytest --nbmake examples/
```

## examples, demos and tutorials

check out the [`examples/` directory](https://github.com/beykyle/rxmc/blob/main/examples/). First install example specific dependencies:

```
pip install -r examples/requirements.txt
```

In particular, the following notebooks are useful for getting started with `rxmc`: 
- [`examples/linear_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/linear_calibration_demo.ipynb) for an illustrative example of fitting a line to data, which serves as the basic `rxmc` tutorial.
- [`examples/30s_optical_potential_calibration.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/30s_optical_potential_calibration.ipynb) for a demo of a full Bayesian calibration of a a local optical potential to real experimental data using `rxmc` and `jitR`, in only 30 seconds!
