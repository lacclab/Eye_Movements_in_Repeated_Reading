<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#eye_movements_in_repeated_reading)

[![Ruff](https://github.com/lacclab/Eye_Movements_in_Repeated_Reading/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/lacclab/Eye_Movements_in_Repeated_Reading/actions/workflows/ruff.yml)

# Déjà Vu: Eye Movements in Repeated Reading
## Yoav Meiri<sup>1</sup>, Yevgeni Berzak<sup>1,2</sup>
### <sup>1</sup>Faculty of Data and Decision Sciences, Technion - Israel Institute of Technology, Haifa, Israel <br> <sup>2</sup>Department of Brain and Cognitive Sciences, Massachusetts Institute of Technology, Cambridge, USA

This repository contains all the code used in the paper "Déjà Vu: Eye Movements in Repeated Reading" (CogSci2024) by Yoav Meiri, Yevgeni Berzak

Contact: meiri.yoav@campus.technion.ac.il



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#pipeline-reproduction)

## ➤ Pipeline Reproduction

### Data Preprocessing

We utilize OneStop, a broad-coverage corpus of eye movements in reading, collected using an EyeLink 1000 Plus eye tracker (SR Research) at a sampling rate of 1000Hz. The dataset includes 360 L1 English participants reading materials from the OneStopQA corpus.

In the `src/process_IA_rep_for_reread_analysis` directory, the main function `process_df_for_reread_analysis` receives an SR interest area report as input and performs the following:

- Excludes interest areas following common practice (beginning/end of paragraph, beginning/end of line, words with numbers, punctuation, etc.)
- Adds multiple word-level features that will be used for the analysis presented in the paper.

This function is executed at the beginning of each of the three notebooks in the `src` directory.

### Graphs and Statistical Analyses

The `src` folder contains three Jupyter notebooks, one for each section of the paper. Each notebook includes all the code for plots and statistical tests (including the plots and tests themselves). All numbers presented in the paper were directly taken from these notebooks. Each notebook is organized into sub-sections, with each sub-section corresponding to an analysis in the paper. 

The files `julia_linear_mm` and `calc_means_main` contain Python code wrapped around Julia code, which was used for all statistical tests in the paper. The core component in these scripts is the `julia_linear_mm/run_linear_mm` function. This function fits a linear mixed model using the Julia package MixedModels.jl, where all model properties (formula, random effects, linking function, coding system, etc.) are provided using Python objects.

All the mixed-models analyses in this repository were enabled by the `MixedModels.j` (v4.22.2) package in `Julia`


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#prerequisites)

## ➤ Prerequisites

- [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or Conda



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#setup)

## ➤ Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone https://github.com/lacclab/Eye_Movements_in_Repeated_Reading
    cd Eye_Movements_in_Repeated_Reading
    ```

2. **Create a Virtual Environment**

    Create a new virtual environment using Mamba (or Conda) and install the dependencies:

    ```bash
    mamba env create -f environment.yaml
    ```
3. **Install Necessary Packages**

    Install the necessary Julia packages for running linear mixed models:

    ```bash
    python -c 'from juliacall import Main as jl; jl.seval("""import Pkg; Pkg.add("MixedModels"); Pkg.add("DataFrames"); Pkg.add("Distributions")""")'
    ```

4. **Get the Data**
    There is not yet a public version of the data. Hopefully, there will be one in the following months!

5. **Rerunning The Code**
    As stated, each one of the 3 notebooks contains all of the code needed to reproduce the results of the section it belongs to.
