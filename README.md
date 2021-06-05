

Using Interpretable Deep Learning to Model Cancer Dependencies
==============================

The code repository of [Using Interpretable Deep Learning to Model Cancer Dependencies. C.H. Lin, O. Lichtarge
*Bioinformatics*, 2021.](https://doi.org/10.1093/bioinformatics/btab137)
If you have any questions or comments, feel free to contact Jack Chih-Hsu Lin (lin.chihhsu[at]gmail[dot]com).

--------
## Content
 - [Download code](#download-code)
 - [Installation and download data](#installation-and-download-data)
 - [Run experiments](#run-experiments)
 - [Project organization](#project-organization)

--------
## Download code
```bash
git clone https://github.com/LichtargeLab/BioVNN.git
```

--------
## Installation and download data

### Requirements
- [Anaconda](https://www.anaconda.com/) or [MiniConda](https://conda.io/miniconda.html)
- GPU >= 3GB
- Python 3.6.5
- PyTorch >= 1.2.0
- Please see `environment.yml` for more requirements 

### Install environment and download data
```bash
cd BioVNN
./install.sh
```

--------
## Run experiments
### 0. Recommended resource
#### GPU Memory >=3GB is recommended 
#### Currently it only supports GPU and CUDA


### 1. Activate environment
```bash
conda activate BioVNN
```
If it's activated, you will see `(BioVNN)`  at the beginning of your command prompt  

### 2. Example of running 5-fold cross-validation 
```bash
cd src
./run_cv.sh
```

### 3. Example of running time-stamped experiment
- It is required to complete one cross-validation experiment before running the time-stamped experiment.
- Modify the parameter file `params/timestamped.yml`
- Make the `load_result_dir_name=${the directory name of cross-validation result}`
For example: `load_result_dir_name: 20201008201106_clh_v1_19Q3_rna_ep200_ES_p2_SS_ComF_l2_ce_Reactome_ref_PANC`
```bash
cd src
./run_ts.sh
```


--------

Project organization
------------
    BioVNN/
    ├── README.md               <- This document.
    ├── install.sh              <- The script to set up environment and download data.
    ├── environment.yml         <- Conda environment file of package requirement.
    └── src/                    <- Source code.
         ├── run_cv.py          <- The script to run 5-fold cross-validation of BioVNN.
         ├── run_cv_rg.py       <- The script to run 5-fold cross-validation of random group model.
         ├── run_cv_fc.py       <- The script to run 5-fold cross-validation of fully connected network.
         ├── run_ts.py          <- The script to run time-stamped experiments for BioVNN.
         ├── paths.py           <- The script to load environment variables.
         ├── biovnn_model.py    <- The class of BioVNN model.
         ├── dependency.py      <- The class of 5-fold cross-validation.
         ├── timestamped.py     <- The class of time-stamped experiment.
         ├── pytorch_layer.py   <- The class of PyTorch layers and dataloaders.
         ├── utils.py           <- The script of utility functions.
         └── set_logging.py     <- The script to set up log.
    

