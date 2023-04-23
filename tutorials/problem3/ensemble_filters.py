class ensemble_filters():
    
    """
    This class includes two separate smoothing algorithm:
        EnKF: an empirical ensemble Kalman filter
        EnTF: an adaptive ensemble transport filter
    """
    
    
    def EnKF(self, x, y, y_obs):
        
        """
        This function implements the empirical ensemble Kalman filter update.
        
        Parameters
        ----------
        x: states
             A N-by-D array of system states.
        y: observation simulations
             A N-by-O array of simulated observations.
        y_obs: observations
             A O-by-1 vector of observations.
    
        Returns
        -------
        x: states
             A N-by-D array of updated system states.
        """
        
        import numpy as np
        import copy
        
        # Assert that all variables have the correct shape
        if len(x.shape) < 2: raise ValueError("x must be two-dimensional. Current shape: "+str(x.shape))
        if len(y.shape) < 2: raise ValueError("y must be two-dimensional. Current shape: "+str(y.shape))
        if len(y_obs.shape) < 2: raise ValueError("y_obs must be a column vector. Current shape: "+str(y.shape))
        if x.shape[0] != y.shape[0]: raise ValueError("x and y must have the same number of rows. Currently, they have "+str(x.shape[0])+" and "+str(y.shape[0])+" rows.")
        if y.shape[1] != y_obs.shape[0]: raise ValueError("y must have as many columns as y_obs has rows. Currently, they have "+str(y.shape[1])+" and "+str(y_obs.shape[0])+".")
        
        # How many samples do we have?
        N           = x.shape[0]
        
        # The EnKF actually needs x and y with N in the columns. We only require
        # the form above for consistency with the transport filters. Hence, let
        # us transport both matrices
        x           = copy.copy(x).T
        y           = copy.copy(y).T
        
        # This is a projector which subtracts the ensemble mean
        projector   = np.identity(N) - np.ones((N,N))/N
        
        # Calculate the anomalies of the current state
        Ax          = x@projector
        
        # Calculate the anomalies of the observation predictions
        Ay          = y@projector
        
        # Calculate the Kalman gain
        gain        = Ax@Ay.T@np.linalg.inv(Ay@Ay.T)
        
        # Apply the EnKF update
        x           = x - gain @ (y - y_obs)
        
        # Undo the transposition of the states
        x           = x.T
        
        return x
        
    def EnTF(self, x, y, y_obs, reduce_states = True, reduce_observations = True):
        
        """
        This function implements an adaptive ensemble transport filter.
        
        Parameters
        ----------
        x: states
             A N-by-D array of system states.
        y: observation simulations
             A N-by-O array of simulated observations.
        y_obs: observations
             A O-by-1 vector of observations.
    
        Returns
        -------
        x: states
             A N-by-D array of updated system states.
        """
        
        import numpy as np
        import copy
        from sklearn.decomposition import PCA
        from transport_map_plus_adapt_44 import transport_map
        
        # Assert that all variables have the correct shape
        if len(x.shape) < 2: raise ValueError("x must be two-dimensional. Current shape: "+str(x.shape))
        if len(y.shape) < 2: raise ValueError("y must be two-dimensional. Current shape: "+str(y.shape))
        if len(y_obs.shape) < 2: raise ValueError("y_obs must be a column vector. Current shape: "+str(y.shape))
        if x.shape[0] != y.shape[0]: raise ValueError("x and y must have the same number of rows. Currently, they have "+str(x.shape[0])+" and "+str(y.shape[0])+" rows.")
        if y.shape[1] != y_obs.shape[0]: raise ValueError("y must have as many columns as y_obs has rows. Currently, they have "+str(y.shape[1])+" and "+str(y_obs.shape[0])+".")
        
        # How many samples do we have?
        N           = x.shape[0]
        
        # Create local copies of x and y
        x           = copy.copy(x)
        y           = copy.copy(y)
        
        # Ensemble transport filters can become computationally expensive in 
        # high-dimensional settings. To increase efficiency, we apply a 
        # principal component reduction to both states and observations.
        if reduce_states:
            pca_x   = PCA(n_components = 0.95)
            pca_x   .fit(x)
            x       = pca_x.transform(x)
        if reduce_observations:
            pca_y   = PCA(n_components = 0.95)
            pca_y   .fit(y)
            y       = pca_y.transform(y)
            y_obs   = pca_y.transform(y_obs.T).T
            
        # Concatenate the traning samples
        map_input   = np.column_stack((y,x))
            
        # Create the transport map object
        tm  = transport_map(
            X               = map_input,
            adaptation      = True,
            polynomial_type = "edge-controlled hermite",
            weight_type     = "cubic spline",
            monotonicity    = "separable monotonicity",
            verbose         = True)
        
        # Call the adaptive optimization
        tm.adapt_map(
            max_order_mon       = 3, # Limit complexity to third-order polynomials
            max_order_nonmon    = 3, # Limit complexity to third-order polynomials
            bincount            = "auto", # automatically find the bincount
            sequential_updates  = False, # Do not update the map components one at a time (slower)
            limit_to_mon        = False, # Do not restrict off-diagonal entries to the complexity of their respective diagonal
            Gaussianity_error   = 0.05, # Acceptable rate of type-1 errors in the Gaussianity test
            independence_error_ratio = 0.5, # Empirical rate between type-1 and type-2 errors for the independence test
            independence_threshold = 0.) # Only uncorrelated (r=0.) samples count as independent
        
        # Obtain the pushforward samples
        z   = tm.map(map_input)
        
        # Conditionally invert the map
        x   = tm.inverse_map(
            X_star  = np.repeat(y_obs.T,axis=0,repeats=N), # Observed values
            Z       = z[:,y.shape[-1]:])[:,y.shape[-1]:] # Pushforward samples
        
        # Revert the dimension reduction
        if reduce_states:
            x       = pca_x.inverse_transform(x)
        
        return x