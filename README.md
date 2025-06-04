# rxmc
Uncertainty quantification and calibration of reaction models with Markov-Chain Monte Carlo, with flexible and composable models for the likelihood and corpus of constraints.
- Curate a corpus of experimental constraints using [`exfor_tools`](https://github.com/beykyle/exfor_tools)
- Efficiently calculate corresponding observables using [`jitR`](https://github.com/beykyle/jitr)
- Choose from a variety of likelihood models, or extend the basic [`Constraint`]() class to implement your own.

An example of this code in use is in the development of the [East Lansing Model](https://github.com/beykyle/elm)

## installation

### for development
To modify the model, first clone and build
```bash
git clone git@github.com:beykyle/rxmc.git
cd rxmc
python3 -m build
```

Then install an editable version locally like so:

```
pip install -ve .
```

Note that `pip` will install package dependencies listed in `requirements.txt`. It is **highly recommended** that you use an isolated virtual environment (e.g. using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda/mamba](https://mamba.readthedocs.io/en/latest/)), as this action will install all of the dependencies in `requirements.txt`, at the specific version required.

If you don't want to create an isolated environment for `rxmc`, but also don't want `pip` to overwrite the package versions you have with the ones in `requirements.txt`, you can

```
pip install -ve --no-deps .
```
This will require that your current python environment satisfies the `requirements.txt`. 

## test

```
python -m unittest discover ./test
```
## examples

Check out [`examples/linear_calibration_demo.ipynb`](https://github.com/beykyle/rxmc/tree/main/examples) for an illustrative example of fitting a line to data with various likelihood models.

Other examples demonstrating actual reaction model fitting coming soon.

For a fully fledged example of this code for calibrating an uncertainty quantified global optical potential, check out [East Lansing Model](https://github.com/beykyle/elm)
