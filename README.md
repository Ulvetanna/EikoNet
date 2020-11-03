
# EikoNet: A deep neural networking approach for seismic ray tracing
---


![GitHub last commit](https://img.shields.io/github/last-commit/ulvetanna/EikoNet?style=plastic)
![GitHub stars](https://img.shields.io/github/stars/ulvetanna/EikoNet?style=social)
![GitHub forks](https://img.shields.io/github/forks/ulvetanna/EikoNet?style=social)
![GitHub followers](https://img.shields.io/github/followers/ulvetanna?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ulvetanna/EikoNet?style=social)

<!---
Get this done soon
![PyPI](https://img.shields.io/pypi/v/EikoNet?style=plastic)
![Conda](https://img.shields.io/conda/v/ulvetanna/EikoNet?style=plastic)
![PyPI - License](https://img.shields.io/pypi/l/EikoNet?style=plastic)
![Conda](https://img.shields.io/conda/dn/ulvetanna/EikoNet?style=plastic)
--->


***
## Introduction

EikoNet is a software package that allows the training of a neural network that satisfies the Factored Eikonal
for the computation of travel-time from any source-reciever pair in a user defined velocity model.

This approach is outline in greater detail in the publication:
Smith et al. (2020) - EikoNet: A deep neural networking approach for seismic ray tracing - [link to paper](https://arxiv.org/pdf/2004.00361.pdf)


## Colab Jupyter Notebook
We have provided a Colab notebook to allow uses to go through the examples outlined in the paper provided above. The Colab notebook is separated into a series of sections which are all standalone executable scripts, but require the download and build of the software given in the 'Introduction' section of the notebook. As the software develops we will include additional sections.

Link to the Colab can be found at: [link](https://colab.research.google.com/drive/1Hlz2bJQ1bwIVkFtWnBmZIaLR-kbCk_TQ?usp=sharing)


## Installation

The EikoNet software can be installed by using

```
  python setup.py install
```

If you wish to also plot comparisons with the python finite-difference software scikit-fmm ([link](https://pypi.org/project/scikit-fmm/0.0.7/)) then an additional packge is required.
```
  pip install scikit-fmm
```

## Guide to Training and Setup
The software can be separated into the imports:
```
from EikoNet import database as db
from EikoNet import model as md
from EikoNet import plot as pt
```
Outlined below is some background information to each of these sections. Additional information and usage can be found at the Colab [link](https://colab.research.google.com/drive/1Hlz2bJQ1bwIVkFtWnBmZIaLR-kbCk_TQ?usp=sharing).


* EikoNet database
  * This section contains the sampling methods and velocity model classes used in the training of the EikoNet. Outlined below are some of the key information of these functions and setting up your own velocity model class.
  * ```db.Database``` - Function called during training to create a random source-receiver database for a given velocity class. Inputs are a input velocity function with the optional arguments: `create` if you wish to create a new database on load an existing at path, `Numsamples` the number of samples to draw, `randomDist` whether to use a random distance sampling over the standard random location method.

  * ```db.ToyProblem_Homogeneous``` - A class describing how random sampling of point in the model space relates to the velocity. Inside this class you will have a `__init__` that must contains the the `xmin` and `xmax` for the model dimension to sample. The function `eval` takes in an array of source receiver pairs and returns the observed velocity at the source and receiver locations.

  * The classes ```db.ToyProblem_Homogeneous```,```db.ToyProblem_1DGraded```,```db.ToyProblem_BlockModel```,```db.ToyProblem_Checkerboard``` and ```db.Graded1DVelocity``` all represent classes to evaluate for different velocity models. In all cases the class functions must have a `xmin` and `xmax` in the `init`, and a function `eval`. Additional functions and variables are optional depending on the use problem




* EikoNet Model
  * The model class contains all the information about the network architecture, model training , model validation, post training travel-time formulation for new points, post training velocity formulation for new points and stationary point formulation

  * ```md.model``` - Called initially to setup the structure required for the problem. Inputs are a Velocity model class, path and file names, device to run on and additional optional arguments.

  * ```md.model.train``` - Called in training the EikoNet for a specific Velocity model class. This function requires number of epochs to run over, the resampling bounds to run between (typical ```[0.1,0.9]``` representing a clamp between 10-90%) and the optional validation percentage.

  * ```md.model.load``` - Loading a pre-trained EikoNet model. Input is the path to the eikonet model.

  * ```md.model.TravelTime``` - Takes in a numpy array of shape ```[NumPoints,6]``` where the table is the source - receiver points in ```[Xsrc,Ysrc,Zsrc,Xrcv,Yrcv,Zrv]``` format. This function returns the travel-time between each of the source receiver pairs.

  * ```md.model.Velocity``` - Takes in a numpy array of shape ```[NumPoints,6]``` where the table is the source - receiver points in ```[Xsrc,Ysrc,Zsrc,Xrcv,Yrcv,Zrv]``` format. This function returns the velocity at the receiver location.

  * ```md.model.StationayPoints``` - Takes in a two source locations defined in the form ```[Xsrc1,Ysrc1,Zsrc1]``` and ```[Xsrc2,Ysrc2,Zsrc2]``` to try and determine stationary point values between each of these points. You can either compute for a series of random locations, with number specified by ```numPoints```, or by defining the optimal argument ```Xpoints``` which takes an array of size ```[NumPoints,3]``` representing the point locations in space

* EikoNet Plot
  * A plotting class that is able to plot the recovered travel-time, recovered velocity model, observed velocity model and finite-difference travel-times (optional argument). This class function should only be used for the toy velocity model examples. However, the ```md.model.TravelTime``` and ```md.model.Velocity``` could be used with `matplotlib` to evaluate for a user defined plotting


## Developers
Corresponding email - jon_smith83@hotmail.co.uk

Jonathan Smith         - California Institute of Technology\
Kamyar Azizzadenesheli - California Institute of Technology\
Zachary Ross           - California Institute of Technology\
Jack Muir              - California Institute of Technology
