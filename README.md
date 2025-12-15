# rxmc
Bayesian calibration of reaction models with Markov-Chain Monte Carlo, with flexible and composable models for the likelihood and body of evidence.
- curate the corpus of experimental constraints (e.g. using [`exfor_tools`](https://github.com/beykyle/exfor_tools))
- efficiently calculate your model's corresponding predictions for these observables using [`jitR`](https://github.com/beykyle/jitr)
- choose from a variety of likelihood models, or extend the basic [`LikelihoodModel`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/likelihood_model.py) class to implement your own.
- package the constraints, physical model (and solver), and likelihood model together in a [`Evidence`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/evidence.py) object which provides the likelihood of a given model parameter, for use in Bayesian calibration 
- run Bayesian calibration using a [`Walker`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/walker.py)

An example of this code in use is in the development of the [East Lansing Model](https://github.com/beykyle/elm)

Check out the [`examples/` directory](https://github.com/beykyle/rxmc/blob/main/examples/).

## documentation
- TBD

## installation
### pypi

- TBD

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
python -m unittest discover ./test
```
## examples, demos and tutorials

check out the [`examples/` directory](https://github.com/beykyle/rxmc/blob/main/examples/)

In particular, the following notebooks are useful for getting started with `rxmc`: 
- [`examples/linear_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/linear_calibration_demo.ipynb) for an illustrative example of fitting a line to data, which serves as the basic `rxmc` tutorial.
- [`systematic_err_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/systematic_err_demo.ipynb) for a comparison of some of the likelihood models built into `rxmc`, and how to use them for situations involving systematic errors and multiple independent experimental constraints
- [`examples/30s_optical_potential_calibration.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/30s_optical_potential_calibration.ipynb) for a demo of a full Bayesian calibration of a a local optical potential to real experimental data using `rxmc` and `jitR`, in only 30 seconds!

## use with third party MCMC samplers

The `Evidence` class in `rxmc` can be used with any MCMC sampler that requires a function which returns the log-likelihood of a given parameter set. `rxmc.config.CalibrationConfig` provides a convenient way to package together an `Evidence` object with MCMC sampler settings, and can be used to run MCMC sampling with third party samplers like [`emcee`](https://emcee.readthedocs.io/en/stable/) or [`pymc`](https://www.pymc.io/). A fully fledged example of setting up an inference problem
with `rxmc`, and then using an `emcee` `EnsembleSampler` to sample from the posterior in a massively parallel MPI approach can be found in [`examples/emcee/`](https://github.com/beykyle/rxmc/blob/main/examples/emcee/). 
