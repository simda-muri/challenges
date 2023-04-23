# create observation data
def linear_observation_model(x,H,obs_sd,noise="additive"):
    
    """
    This function implements a linear additive or multiplicative observation 
    model.
    
    Parameters
    ----------
    x: states
         A N-by-D array of system states.
    H: observation operator
         A linear D-by-O array.
    obs_sd: observation error standard deviation.
         A non-negative scalar for the observation error standard deviation.

    Returns
    -------
    y: predicted observations
         A N-by-O array of updated system states.
    """
    
    import numpy as np
    import scipy.stats
    import copy
    
    # Create the deterministic observation prediction
    y       = copy.copy(x)@H.T
    
    # Add noise
    if noise == "additive":
        y       += scipy.stats.norm.rvs(
            loc     = 0,
            scale   = obs_sd,
            size    = y.shape)
    elif noise == "multiplicative":
        y       *= scipy.stats.norm.rvs(
            loc     = 1,
            scale   = obs_sd,
            size    = y.shape)
    
    # Truncate any negative ice thickness values
    y[y<0]  = 0 
    
    return y