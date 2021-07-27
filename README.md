# DiffusionMLE

This package provides an efficient maximum likelihood estimator to extract diffusion coefficients from single-molecule tracking experiments subject to static and dynamic noise.  It can either be used to find a single diffusion coefficient that best describes the underlying dynamics of a trajectory sample, or to analyze each trajectory separately.  

In case of heterogeneous data, i.e., data originating from subpopulations with differing diffusion coefficients, we provide an expectation-maximization algorithm that assigns the trajectories to subpopulations in a probabilistic manner.  The optimal number of subpopulations is determined with the help of a quality factor of known analytical distribution. 

For more details on the theoretical framework, please refer to the associated open access publication:
> J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-particle tracking experiments", *Journal of Chemical Physics* **154**, 234105 (2021).  https://doi.org/10.1063/5.0038174

Please cite this reference if you use DiffusionMLE to analyze your data.  


## Installation

Currently, the package is not in a registry.  It must therefore be added by specifying a URL to the repository:
```julia
using Pkg; Pkg.add(url="https://github.com/bio-phys/DiffusionMLE")
```
Users of older software versions may need to wrap the contents of the brackets with `PackageSpec()`.  



## Usage

### Importing data

Each trajectory can be seen as a *d*-dimensional array (`Array{Float64,2}`), so the data set should be of the type `Array{Array{Float64,2},1}`.  

Your data can, e.g., be read in as follows:
```julia
using DelimitedFiles

file_names = readdir("./trajectories/")
data = [ readdlm(file_names[i]) for i = 1 : length(file_names) ]
```
Next to the trajectories, we also need an array of blurring coefficients that should be equal in length to `data`.  If the illumination profile of the shutter is uniform, we can simply use
```julia
B = [1/6 for m = 1 : length(data)]
```
If the code is used to analyze simulation data with perfect time resolution, the blurring coefficients should be set to zero.  



### Homogeneous data

Assuming that the data are homogeneous, we can estimate the parameters `a2` and `σ2` using the `MLE_estimator` function:
```julia
using DiffusionMLE

[a2, σ2] = MLE_estimator(B,data)
```
`MLE_estimator` has an optional argument, `interval`, which is set to `[0.0,10000.0]` by default, but can be varied in the (unlikely) event that the optimizer fails to converge.  Specifying the `interval` can also speed up the evaluation, as seen by comparing the output of the following two lines:
```
@time MLE_estimator(B,data)
@time MLE_estimator(B,data,[0.0,10.0])
```
The parameters `a2` and `σ2` both have dimension length squared, so if the trajectory coordinates are given in nanometers then said parameters have dimension nanometer squared.  The associated diffusion coefficient is, irrespective of the dimension *d*, given by
```julia
D = 0.5*σ2/Δt
```
The `MLE_errors` function provides an estimate of the parameter uncertainties:
```julia
[δa2, δσ2] = MLE_errors(B,data,[a2, σ2])
```
where the uncertainty `δD`, in analogous fashion to `D`, follows from `δσ2` by division.  

A more detailed example can be found in [the examples directory](examples).  



### Heterogeneous data

The code relevant for the analysis of heterogeneous data exploits threading, so it is recommended to run the command `export JULIA_NUM_THREADS=n`, with `n` being the number of available (physical) cores, before launching Julia.  This speeds up the numerics significantly.  

To check the number of available cores for threading, simply run
```julia
using Base.Threads; nthreads()
```
This should print the number `n` if the above-mentioned command was properly executed.  

If the data are heterogeneous, it can be analyzed with the function `global_EM_estimator` as follows:
```julia
estimates, L, T = global_EM_estimator(K=2,N_local=500,N_global=50,a2_range=[0.02,20.],σ2_range=[0.02,20.],B,data)
```
Here, we consider `K=2` subpopulations, and initiate the parameter search `N_global=50` times with randomly chosen parameters from `a2_range` and `σ2_range`.  If a search does not converge within `N_local=500` steps, it is broken off.  The optional argument `tolerance` is set to `1.0e-10` by default, and determines the convergence rate of local parameter searches.  Analogous to `MLE_estimator`, the function `global_EM_estimator` offers the optional argument `interval` that can be set manually to fix possible convergence issues.  

The output includes the likelihood score `L`, an `estimates` array, where the *i*-th column contains the parameter estimates `[a2, σ2, P]` for the *i*-th subpopulation, and the classification coefficients `T`.  The latter can be used to assign the trajectories to subpopulations:
```julia
B_sub, data_sub = sort_trajectories(K=2,T,B,data)
```

A more detailed example can be found in [the examples directory](examples).  


