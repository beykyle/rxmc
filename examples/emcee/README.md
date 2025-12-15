## Massively parallel calibration using the emcee Ensemble Sampler

This directory contains an example of using `rxmc` to set up a calibration task that can be run in a massively parallel fashion using the `emcee` Ensemble Sampler. The example used here is a calibration of a simple polynomial model to synthetic data, but the same approach can be used for any model supported by `rxmc`.

This directory contains two python scripts that provide a command line interface for launching massively parallel calibration tasks using the `emcee` Ensemble Sampler. They can be launched, for example, using the slurm script provided in this directory. These are perfectly generic.

Calibration of any `rxmc.evidence.Evidence` object is supported. First, one must define an `rxmc.config.CalibrationConfig` object that specifies the calibration parameters, priors, and other settings. This configuration object must be pickled to a file using `dill`. An example jupyter notebook demonstrating how to create and pickle a `CalibrationConfig` object can be found also in this directory.

To do this demo:

1. Install the required packages listed in `requirements.txt`.
2. Run the Jupyter notebook `create_calibration_config.ipynb` to create and pickle a `CalibrationConfig` object, which will be written to `rxmc_poly_demo_conf.pkl`
3. Launch the slurm script `run_emcee_calibration.slurm` to run the calibration using the `emcee` (e.g. `sbatch run_emcee_calibration.slurm`).

The results of the calibration will be written to the output directory specified in the slurm script, using the emcee HDF5 backend.

For more information about emcee, see the [emcee documentation](https://emcee.readthedocs.io/en/stable/).
