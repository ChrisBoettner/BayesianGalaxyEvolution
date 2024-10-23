# A Bayesian Empirical Galaxy Evolution Model

This project presents an empirical model for the co-evolution of galaxies, supermassive black holes (SMBHs), and dark matter halos across cosmic time (z = 0-10). By combining the evolution of the halo mass function (HMF) with simple analytical prescriptions for baryonic processes, the model reproduces key observations and offers qualitative and quantitative predictions in a computationally efficient manner.

**Paper**: [Link](https://arxiv.org/abs/2303.11368)

## Model Description

The model connects observable baryonic structures (galaxies and SMBHs) to their host dark matter halos using empirical relations. It assumes a one-to-one correspondence between halos and central galaxies/SMBHs, and posits that the observable properties of galaxies and SMBHs are determined by invertible functions of their host halo mass, potentially evolving with redshift.

The model incorporates three feedback regimes:

* **Stellar Feedback Regime:** In low-mass halos, stellar feedback regulates gas content, suppressing star formation and black hole growth.
* **AGN Feedback Regime:** In massive halos, AGN feedback takes over, regulating gas content and growth.
* **Turnover Regime:** This regime represents the transition between stellar and AGN feedback dominance.

Specific functional forms are used to describe the relationship between observable quantities and halo mass, informed by existing observational relations. These include a double power law for stellar mass and UV luminosity, a power law for black hole mass, and an Eddington ratio-dependent relation for quasar bolometric luminosity.  The model incorporates an Eddington Ratio Distribution Function (ERDF) to account for the variation in black hole accretion rates.

## Key Features

* **Bayesian Calibration:** The model is calibrated using a fully Bayesian approach, utilizing observed number density functions (GSMF, UVLF, BHMF, QLF) and MCMC sampling to constrain model parameters and their probability distributions. This allows for a comprehensive understanding of the parameter space and robust uncertainty estimation.
* **Computational Efficiency:** The model's analytical nature makes it computationally efficient, facilitating exploration of the parameter space and extrapolation to higher redshifts.
* **Scatter Exploration (Experimental):** The project includes experimental code to incorporate scatter in the quantity-halo mass relations using various probability distributions (lognormal, normal, Cauchy, skew lognormal).  This feature is still under development and should be used with caution.

## Repository Contents

* **`README.md`:** This file.
* **`environment.yaml`:** Conda environment specification file.  Use `conda env create -f environment.yaml` to create the required environment.
* **`make_plots.py`:** Script to generate and save plots of various model outputs and comparisons with observational data.
* **`make_tables.py`:** Script to generate LaTeX tables summarizing model parameters and survey predictions.
* **`model` directory:**  Contains the core model code, including modules for:
    * **`analysis`:**  Functions for calculations and analysis of model results, including number density percentiles, quantity-halo mass relations, density evolution, reference function fitting, etc.
    * **`calibration`:** Modules for model calibration using least-squares fitting and MCMC sampling.
    * **`data`:**  Functions for loading observational data and HMF functions.
    * **`eddington.py`:** Implements the ERDF class.
    * **`helper.py`:** Utility functions.
    * **`hmf`:**  Modules for working with the halo mass function.
    * **`interface.py`:** Provides high-level functions for loading, saving, and running the model.
    * **`model.py`:** Core model classes and functions.
    * **`physics.py`:**  Implements the different physics models relating observable quantities to halo mass.
    * **`plotting`:** Modules for generating plots of model results.
        * `convience_functions.py`: Helper functions for plotting, including data loading and formatting.
        * `plotting.py`: Core plotting functions for various model outputs.
        * `presentation_plots.py`: Specialized plotting functions for presentations and illustrations.
        * `settings.rc`: Matplotlib style sheet.
    * **`quantity_options.py`:** Defines quantity-specific parameters and options.
    * **`scatter.py`:** Experimental module for incorporating scatter in the quantity-halo mass relation.
* **`parameter.csv`:**  Example parameter file.
* **`playing_around.py` & `playing_around_scatter.py` & `referee_scatter_playing_around.py`:** Example scripts demonstrating model usage and plotting.
* **`run.py`:**  Command-line interface for running the model.

## Usage

1. **Create the Conda environment:** `conda env create -f environment.yaml`
2. **Activate the environment:** `conda activate empirical-galaxy-model`
3. **Run the model:** `python run.py`.  You will be prompted to specify the physical quantity, physics model, and other options.
4. **Generate plots:** `python make_plots.py`
5. **Generate tables:** `python make_tables.py`

## Example Scripts

The `playing_around.py` and `playing_around_scatter.py` scripts provide examples of loading the model, calculating various quantities, and generating plots.  These scripts can be adapted for your own analysis.


## Data Availability

The code, data used for calibration, and posterior parameter distributions are available on Zenodo: [https://doi.org/10.5281/zenodo.7552484](https://doi.org/10.5281/zenodo.7552484).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
