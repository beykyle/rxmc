# rxmc
Bayesian calibration of reaction models with Markov-Chain Monte Carlo, with flexible and composable models for the likelihood and body of evidence.
- curate the corpus of experimental constraints (e.g. using [`exfor_tools`](https://github.com/beykyle/exfor_tools))
- efficiently calculate your model's corresponding predictions for these observables using [`jitR`](https://github.com/beykyle/jitr)
- choose from a variety of likelihood models, or extend the basic [`LikelihoodModel`](https://github.com/beykyle/rxmc/blob/main/src/rxmc/likelihood_model.py) class to implement your own.
- package the constraints, physical model (and solver), and likelihood model together in a `rxmc.evidence.Evidence` object which provides the likelihood of a given model parameter, for use in Bayesian calibration 
- run Bayesian calibration using an `rxmc.Walker`

An example of this code in use is in the development of the [East Lansing Model](https://github.com/beykyle/elm)

See `examples/` for a variety of demos.

## documentation
- TBD

## installation
### pypi

Not yet available

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
## examples

Check out the demos: 
- [`examples/linear_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/linear_calibration_demo.ipynb) for an illustrative example of fitting a line to data with various likelihood models.
- [`systematic_err_demo.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/systematic_err_demo.ipynb) for a demonstration of the likelihood models built into `rxmc`, and how to use them for situations involving systematic errors and multiple independent experimental constraints
- [`fitting_an_optical_potential.ipynb`](https://github.com/beykyle/rxmc/blob/main/examples/fitting_an_optical_potential.ipynb) for a demonstration of how to fit a local optical potential to experimental data using `rxmc` and `jitR`. 

Other examples demonstrating actual reaction model fitting coming soon.

For a fully fledged example of this code for calibrating an uncertainty quantified global optical potential, check out [East Lansing Model](https://github.com/beykyle/elm)
