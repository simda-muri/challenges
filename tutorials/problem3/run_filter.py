import numpy as np
import matplotlib.pyplot as plt
from forecast_model import forecast
from observation_model import linear_observation_model
from ensemble_filters import ensemble_filters
import scipy.stats
import os
import copy

# Out sea ice domain is circular. To sample circular noise efficiently, we need
# a circular convolution. This is what this function is for
def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

# Get a colormap for later plotting
cmap    = plt.get_cmap("turbo")

# Set a random seed
np.random.seed(0)

# Let's set some parameters
dx          = 1E4   # Spatial resolution
dt          = 360   # Temporal resolution
T           = 365   # Number of time steps
N           = 1000  # Ensemble size
D           = 200   # State dimension
obs_sd      = 0.1   # Observation error standard deviaton
forecast_sd = 0.5   # Forecast error standard deviation

# Create an observation operation
obs_range   = 10 # Make an observation every 10 cells
obspts      = np.arange(0,D,obs_range)+int(obs_range/2) # Arrange it around the domain
H           = np.zeros((len(obspts),D)) # Create an empty observation operator
for row in range(len(obspts)): # Every observation is spatially aggregated
    H[row,:]    = np.exp(-(np.arange(D)-obspts[row])**2/(obs_range*5))
H[H < 0.01] = 0 # Truncate low contributions
H           = np.einsum('ij,i->ij',H,1/np.sum(H,axis=-1)) # Normalize the rows

# How many observations to we make?
O       = H.shape[0]

# =============================================================================
# Generate random wind
# =============================================================================

# The wind focing is circular and random
filter_scale = 100 # Size of the convolution filter
correlation_scale = 500 # Spatial correlation of wind noise

# Generate the convolution filter kernel
x = np.arange(-filter_scale, filter_scale)
y = np.arange(-filter_scale, filter_scale)
X, Y = np.meshgrid(x, y)
dist = np.sqrt(X*X + Y*Y)
filter_kernel = np.exp(-dist**2/(2*correlation_scale))

# Sample raw noise as a basis for the wind
wind = np.random.randn(D,T) 

# Apply the 2D convolution
#   first dimension: space
#   second dimension: time
wind = scipy.signal.convolve2d(wind, filter_kernel, mode='same', boundary='wrap')

# Normalize the wind between -10 and 10 m/s
wind    -= np.min(wind)
wind    /= np.max(wind)
wind    = wind*2-1
wind    *= 10

# =============================================================================
# Generate initial conditions
# =============================================================================

# Create the initial state vector
h       = np.zeros((D)) + 0.2
h[:40]  = 2
h[-40:] = 2

# Let's simulate the true sea ice dynamics
hs_true     = [h]
for t in np.arange(1,T,1):
    new_h   = forecast(
        u       = wind[:,t],
        h       = hs_true[-1],
        dx      = dx,
        dt      = dt)
    hs_true.append(new_h)
hs_true     = np.asarray(hs_true).T

# Generate synthetic observations
Y       = []
for t in range(T):
    Y   .append(
        linear_observation_model(
            x       = hs_true[:,t][np.newaxis,:],
            H       = H,
            obs_sd  = obs_sd,
            noise   = "additive"))
Y       = np.asarray(Y).T


# Pre-allocate a variable for the states
hs  = np.zeros((N,D,T))

# Draw the initial states 
noise       = np.random.randn(N, D)
kernel      = np.asarray([np.exp(-x**2/2) for x in np.linspace(-10,10,D)])
for n in range(N):
    noise[n,:]       = conv_circ( noise[n,:], kernel )
    noise[n,:]  -= np.min(noise[n,:])
    noise[n,:]  /= np.max(noise[n,:])
    noise[n,:]  *= 3
    noise[n,:]  += 0.25
hs[:,:,0]   = copy.copy(noise)

# %%

# =============================================================================
# Start filtering
# =============================================================================

# Initiate an array for simulated observations
Y_sim       = np.zeros((N,O,T))

# Shall we plot the output?
plot_results= True

# Initiate an ensemble filter object
filters     = ensemble_filters()

filter_mode = "EnKF" # alternative: "EnTF"

# Let's start filtering
for t in np.arange(0,T,1):
    
    print("Filtering "+str(t))
    
    # Step 1: implement the EnKF update ---------------------------------------
    
    # Generate observations
    Y_sim[:,:,t]    = linear_observation_model(
        x       = hs[:,:,t],
        H       = H,
        obs_sd  = obs_sd,
        noise   = "additive")
    
    # apply the filter update
    if filter_mode == "EnKF": # Apply an EnKF update
        hs[:,:,t]   = filters.EnKF(
            x       = hs[:,:,t],
            y       = Y_sim[:,:,t],
            y_obs   = Y[:,:,t])
    else: # Apply a nonlinear adaptive EnTF update
        hs[:,:,t]   = filters.EnTF(
            x       = hs[:,:,t],
            y       = Y_sim[:,:,t],
            y_obs   = Y[:,:,t])
    
    # Prevent physically impossible states
    hs[:,:,t][hs[:,:,t] < 0] = 0
    
    # Step 2: Forecast to the next timestep -----------------------------------
    
    if t < T-1:
        
        # Sample circular noise
        noise       = np.random.randn(N, D)
        kernel      = np.asarray([np.exp(-x**2/2) for x in np.linspace(-10,10,D)])
        for n in range(N):
            noise[n,:]       = conv_circ( noise[n,:], kernel )
            noise[n,:]  -= np.min(noise[n,:])
            noise[n,:]  /= np.max(noise[n,:])
            noise[n,:]  = noise[n,:]*2 - 1
            noise[n,:]  *= forecast_sd
    
        # Make a forecast
        hs[:,:,t+1] = forecast(
            u       = np.repeat(wind[:,t][np.newaxis,:],axis=0,repeats=N).flatten(),
            h       = hs[:,:,t].flatten(),
            dx      = dx,
            dt      = dt,
            noise   = noise.flatten()).reshape((N,D))
    
    # Step 3: Plot results
    if plot_results:
    
        # Extract quantiles
        q05     = np.quantile(hs[:,:,t], q = 0.05, axis = 0)
        q25     = np.quantile(hs[:,:,t], q = 0.25, axis = 0)
        q50     = np.quantile(hs[:,:,t], q = 0.50, axis = 0)
        q75     = np.quantile(hs[:,:,t], q = 0.75, axis = 0)
        q95     = np.quantile(hs[:,:,t], q = 0.95, axis = 0)
    
        # Plot results
        plt.close("all")
        plt.figure(figsize=(12,8))
        
        from matplotlib.gridspec import GridSpec
        gs  = GridSpec(
            nrows   = 2,
            ncols   = 1,
            height_ratios = [1,0.1])
        
        plt.subplot(gs[0])
        
        plt.title("timestep "+str(t).zfill(3)+" | "+str(T).zfill(3))
        
        plt.plot(
            np.linspace(0,2000,D),
            hs_true[:,t],
            color   = "r",
            ls      = "--",
            label   = "synthetic truth",
            zorder  = 5)
        
        xpts    = np.linspace(0,2000,D)
        dxpts   = np.diff(xpts[:2])
        for o in range(O):
            
            for idx,xpt in enumerate(xpts):
            
                if H[o,idx] != 0:
                    plt.plot(
                        [xpt - dxpts/4,xpt + dxpts/4],
                        [Y[o,0,t],Y[o,0,t]],
                        alpha   = H[o,idx]/np.max(H[o,:]),
                        color   = "xkcd:cerulean",
                        zorder  = 10,
                        lw      = 5)
            
        plt.fill(
            list(np.linspace(0,2000,D))+list(np.flip(np.linspace(0,2000,D))),
            list(q05) + list(np.flip(q95)),
            color   = "xkcd:silver",
            label   = "5% - 95% quantile",
            zorder  = 1)
        
        plt.fill(
            list(np.linspace(0,2000,D))+list(np.flip(np.linspace(0,2000,D))),
            list(q25) + list(np.flip(q75)),
            color   = "xkcd:grey",
            label   = "25% - 75% quantile",
            zorder  = 2)
        
        plt.plot(
            np.linspace(0,2000,D),
            q50,
            color   = "xkcd:dark grey",
            label   = "50% quantile",
            zorder  = 3)
        
        plt.xlabel("distance $x$")
        plt.ylabel("Sea ice thickess $h$")
        
        plt.gca().invert_yaxis()
        plt.gca().set_ylim([5,-0.5])
        
        # Draw an observation line for the legend only
        plt.plot(
            [50,51],
            [-100,-100],
            alpha   = 0.75,
            color   = "xkcd:cerulean",
            label   = "ovservation",
            lw      = 5)
        
        plt.legend(loc="lower center", frameon = False)
        
        xlims = plt.gca().get_xlim()

        plt.subplot(gs[1])
        
        plt.gca().set_xlim(xlims)
        
        arrowpts = (np.arange(0,2000,200)+100)
        
        for pt in arrowpts:
            
            idx = int(pt*D/2000)
        
            height  = 0.4
            
            xpos    = pt
            ypos    = 0
            
            arrowwidth = 0.025*2000
            hscale  = 1
            
            arrow   =  np.column_stack((
                np.asarray([
                    xpos, 
                    xpos + wind[idx,t]/10*100*hscale]+\
                 list(xpos + wind[idx,t]/10*100*hscale + np.sign(wind[idx,t])*np.asarray([0., arrowwidth, 0.]))+\
                [xpos + wind[idx,t]/10*100*hscale, xpos]),
                np.asarray([ypos + height, ypos + height]+list(ypos + np.asarray([1., 0., -1.])*height*2)+[ypos - height, ypos - height]) ))
            
            plt.fill(
                arrow[:,0],
                arrow[:,1],
                color   = cmap(np.abs(wind[idx,t]/10)))
    
        # Remove all axis ticks
        plt.tick_params(left=False,
                        labelleft=False)
        plt.ylabel("wind direction")
        plt.xlabel("distance $x$")
    
        plt.savefig("DA_step_"+str(t).zfill(3)+".png",dpi=300)


