# BrainGrowth

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Configuration of Parameters and Run](#run)

## Introduction

This repository contains the work during the PhD of Xiaoyu WANG, titled "Brain development analysis using MRI and physical modeling".

## Requirements

All scripts were coded in `python 2.7`, but we are working to be compatible to `python 3.7`.

<details>
<summary><b>Python packages and versions</b></summary>

- enum34==1.1.6
- funcsigs==1.0.2
- llvmlite==0.24.0
- nibabel==2.3.1
- numba==0.39.0
- numpy==1.16.2
- numpy-stl==2.10.1
- python-utils==2.3.0
- scikit-learn==0.20.3
- scipy==1.2.1
- singledispatch==3.4.0.3
- six==1.11.0
- Vapory==0.1.1
- mpmath==1.0.0
- os
</details>

The file called `requirements.txt` helps to install all the python libraries.

- Using pip:
```
pip install -r requirements.txt
```

- Using anaconda:
```
conda install --file requirements.txt
```

## Configuration of Parameters and Run

### simulation.py

In simulation.py, there are certain parameters should be set manually:

**PATH_DIR**: Path of output

**mesh_path**: Path of input

**THICKNESS_CORTEX**: Individual cortical thickness

**GROWTH_RELATIVE**: Tangential growth rate

**a**: Average mesh spacing

For sphere5.mesh:

**a**: 0.01

**dt**: 0.05*np.sqrt(rho*a*a/K)

For prm001_30w_Rwhite_petit_taille_2.mesh:

**a**: 0.001

**dt**: 0.01*np.sqrt(rho*a*a/K)

In output.py and in the function writePov, we should change "vertices[:,:] = Ut[SN[:],:]*zoom_pos" to "vertices[:,:] = -Ut[SN[:],:]*zoom_pos" in order to output from the right perspective.
