# DiffusionMLE

This package provides an efficient maximum likelihood estimator to extract diffusion coefficients from single-molecule tracking experiments subject to static and dynamic noise.  It is composed of two modules, <code>DiffusionMLE</code> and <code>DiffusionEM</code>, where the former can be used to analyze trajectories arising from a sample of particles that all adhere to the same diffusive dynamics.  If the data originate from subpopulations with differing diffusion coefficients, the latter provides an expectation-maximization algorithm to sort the trajectories according to their dynamics.  

Note that <code>DiffusionEM</code> exploits threading, so it is recommended to run the command <code>export JULIA_NUM_THREADS=n</code>, with <code>n</code> being the number of available (physical) cores, before launching Julia.  

For more details on the theoretical framework, please refer to the associated publication:
> J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-molecule tracking experiments", https://arxiv.org/abs/2011.09955



## Data format for input

Each trajectory should be a $d$-dimensional array (<code>Array{Float64,2}</code>), so we expect the data set to be of the type <code>Array{Array{Float64,2},1}</code>.  