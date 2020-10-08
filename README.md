# BrainGrowth

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Configuration of Parameters and Run](#run)

## Introduction

This repository contains the work during the PhD of Xiaoyu WANG, titled "Brain development analysis using MRI and physical modeling".

## Requirements

All scripts were coded in `python 2.7`, but they are compatible to `python 3.7` (tested on version 3.7.3).

<details>
<summary><b>Python packages and versions</b></summary>

- enum34==1.1.6
- funcsigs==1.0.2
- llvmlite==0.24.0
- nibabel==2.5.1
- numba==0.39.0
- numpy==1.17.2
- numpy-stl==2.10.1
- python-utils==2.3.0
- scikit-learn==0.21.3
- scipy==1.3.1
- singledispatch==3.4.0.3
- six==1.12.0
- Vapory==0.1.1
- mpmath==1.0.0
- os
- trimesh
- slam
</details>

The file called `requirements.txt` helps to install all the python libraries.

- Using pip:
```
pip install -r requirements.txt
```

## Configuration of Parameters and Run

### simulation.py

Example of dynamic simulations of a neo-Hookean solid with a tangential differential growth.

```
python simulation.py -i './data/sphere5.mesh' -o './res/sphere5' -hc 'whole' -t 0.042 -g 1.829 -gm 'global'
```

**i**: path of input maillage

**o**: path of output

**hc**: whole or half geometry

**t**: cortical thickness

**g**: relative growth rate

**gm**: global or regional growth

### Running a demo

In simulation.py, there are certain parameters should be set manually:

**a**: average mesh spacing

For sphere5.mesh:

**a** = 0.01

#### Input mesh image， output dynamic process
<img src="./docs/B0.png" width = "430px" /><img src="./docs/sphere5.gif" width = "430px" />
