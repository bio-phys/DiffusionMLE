# Code for estimating translational diffusion coefficients from experimental single-particle tracking data
# version 1.0 (15/12/2020)
# Jakob Tómas Bullerjahn (jabuller@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the associated publication: 
# J. T. Bullerjahn and G. Hummer, "Maximum likelihood estimates of diffusion coefficients from single-molecule tracking experiments", https://arxiv.org/abs/2011.09955



module DiffusionMLE

export 
        # MLE:
        MLE_estimator, 
        MLE_errors, 

        # Q-factor analysis:
        Q_factor_analysis, 
        Kuiper_statistic!, 
        Kuiper_p_value, 

        # EM-algorithm:
        local_EM_estimator!, 
        global_EM_estimator, 
        sort_trajectories, 
        subpopulation_analysis

using Base.Threads
using Distributions
using LinearAlgebra
using Optim
using ProgressMeter

include("basic_functions.jl")
include("mle_functions.jl")
include("qfa_functions.jl")
include("em_functions.jl")

end # module
