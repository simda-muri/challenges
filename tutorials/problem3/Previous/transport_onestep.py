#!/usr/bin/env python
# coding: utf-8

"""
This module contains function to solve transport equation 
  dh/dt + d(uh)/dx =0
for ice thickness h for one single time step, using WENO5
spatial discretization technique and TVD RK3 time integration
technique.
"""

import numpy as np
from WENO import *

def Lh_WENOp(u_new,h_old,dx):
    """
    This function computes the approximation of -d(uh)/dx*dx by WENO
    approximation, assuming periodic boundary conditions. 

    Parameters
    ----------
    u_new: numpy.ndarray
         The velocity vector at current time step.
    h_old: numpy.ndarray
         The ice thickness vector at previous time step.
    dx: float
         The mesh size.

    Returns
    -------
    numpy.ndarray
         The WENO approximation of -d(uh)/dx*dx.
    """
    # flux
    flux=np.multiply(u_new,h_old)

    # Lax-Friedrichs Flux Splitting
    alpha = np.max(abs(u_new))
    fp=(flux+alpha*h_old)/2
    fm=(flux-alpha*h_old)/2
    
    # extend fp and use wenol, assume periodic
    fp_temp1=fp[-3:]
    fp_temp2=fp[:2]
    fp_temp=np.concatenate((fp_temp1,fp,fp_temp2))
    
    dfp=dsol_wenol(fp_temp,dx)
    
    # extend fm and use wenor, assume periodic
    fm_temp1=fm[-2:]
    fm_temp2=fm[:3]
    fm_temp=np.concatenate((fm_temp1,fm,fm_temp2))
    
    dfm=dsol_wenor(fm_temp,dx)

    return -(dfp+dfm)


def _transport(u_new,h_old,dx,dt):
    """
    This function computes the ice thickness at current time step
    by solving its corresponding transport equation.

    Parameters
    ----------
    u_new: numpy.ndarray
         The velocity vector at current time step.
    h_old: numpy.ndarray
         The ice thickness vector at previous time step.
    dx: float
         The mesh size.
    dt: flot
         The step size.

    Returns
    -------
    numpy.ndarray
         The ice thickness vector at current time step.
    """

    hdF1=Lh_WENOp(u_new,h_old,dx);
    h_temp1=h_old+dt*hdF1;    
    hdF2=Lh_WENOp(u_new,h_temp1,dx);
    h_temp2=(3*h_old+h_temp1+dt*hdF2)/4;
    hdF3=Lh_WENOp(u_new,h_temp2,dx);
    h_new=(h_old+2*(h_temp2+dt*hdF3))/3; 

    return h_new






