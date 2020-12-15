# DiffusionMLE

This package provides an efficient maximum likelihood estimator to extract diffusion coefficients from single-molecule tracking experiments subject to static and dynamic noise.  It can either be used to find a single diffusion coefficient that best describes the underlying dynamics of a trajectory sample, or to analyze each trajectory separately.  

In case of heterogeneous data, i.e., data originating from subpopulations with differing diffusion coefficients, we provide an expectation-maximization algorithm that assigns the trajectories to subpopulations in a probabilistic manner.  The optimal number of subpopulations is determined with the help of a quality factor of known analytical distribution. 

The code relevant for the analysis of heterogeneous data exploits threading, so it is recommended to run the command <code>export JULIA_NUM_THREADS=n</code>, with <code>n</code> being the number of available (physical) cores, before launching Julia.  This speeds up the numerics significantly.  

For more details on the theoretical framework, please refer to the associated preprint:
> J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-molecule tracking experiments", https://arxiv.org/abs/2011.09955



## Installation

Currently, the package is not in a registry.  It must therefore be added by specifying a URL to the repository:
```julia
using Pkg; Pkg.clone("https://github.com/bio-phys/DiffusionGLS.jl")
```



## Usage

### Importing data

Each trajectory can be seen as a d-dimensional array (<code>Array{Float64,2}</code>), so the data set should be of the type <code>Array{Array{Float64,2},1}</code>.  

Your data can, e.g., be read in as follows:
```julia
using DelimitedFiles

data = readdlm("trajectories.dat")
```
Next to the trajectories, we also need an array of blurring coefficients that should be equal in length to <code>data</code>.  If the illumination profile of the shutter is uniform, we can simply use
```julia
B = [1/6 for m = 1 : length(data)]
```



### Homogeneous data

Assuming that the data is homogeneous, we can estimate the parameters <code>a2</code> and <code>σ2</code> using the <code>MLE_estimator</code> function
```julia
using DiffusionMLE

[a2, σ2] = MLE_estimator(B,data)
```
Said parameters 

A more detailed example can be found in [the examples directory](examples).  







