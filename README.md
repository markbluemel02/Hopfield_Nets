# Hopfield_Nets
This is a fork of the repo https://github.com/ptolmachev/Hopfield_Nets from the paper Tolmachev and Manton (2020). See also the arxiv paper (https://arxiv.org/abs/2010.01472).

## Environment and packages
All packages needed to run the code are listed in `env.yml`. To create a conda environment with all the necessary packages and dependencies, and to activate the environment, run the following commands:
```bash
conda env create -f env.yml
conda activate infomorphic_env
```

## Changes
We added the infomorphic Hopfield networks to the implemented learning_rules. There are also some changes to the testing and plotting functions. In addition, an implementation of Minimum Probability flow (MPF, see https://arxiv.org/abs/1204.2916) was also added. Requires the correct path to the infomorph_networks repo in learning_rules.py .

## Reproducability
Start in src/simulations.py to get the data for the stability plots. They will be saved to the folder data/custom/... . Then plot the data either with calculate_thresholds.py or see the jupyter notebooks in the main project.



