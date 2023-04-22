# This file contains the functions dsol_wenol and dsol_wenor,
# which takes the input vector sol and spatial discretization
# step dx, then returns WENO approximation results d(sol)/dx
# using left stencils and right stencils respectively.

import numpy as np

# left polynomials
def l_p0(x1,x2,x3,x4,x5):
    return (2*x1-7*x2+11*x3)/6
def l_p1(x1,x2,x3,x4,x5):
    return (-x2+5*x3+2*x4)/6
def l_p2(x1,x2,x3,x4,x5):
    return (2*x3+5*x4-x5)/6

# Smooth Indicators (Beta factors)
def _betas(x1,x2,x3,x4,x5):
    beta_0 = 13/12*(x1-2*x2+x3)**2 + 1/4*(x1-4*x2+3*x3)**2
    beta_1 = 13/12*(x2-2*x3+x4)**2 + 1/4*(x2-x4)**2
    beta_2 = 13/12*(x3-2*x4+x5)**2 + 1/4*(3*x3-4*x4+x5)**2
    return (beta_0,beta_1,beta_2)

# left nonlinear weights
def l_weights(beta_0,beta_1,beta_2):
    d0=1/10
    d1=6/10
    d2=3/10
    eps=1e-6
    alpha_0=d0/((eps+beta_0)**2)
    alpha_1=d1/((eps+beta_1)**2)
    alpha_2=d2/((eps+beta_2)**2)
    alpha_sum=alpha_0+alpha_1+alpha_2
    w0=alpha_0/alpha_sum
    w1=alpha_1/alpha_sum
    w2=alpha_2/alpha_sum
    return (w0,w1,w2)

# left numerical flux
def l_weno(x1,x2,x3,x4,x5):
    (beta_0,beta_1,beta_2)=_betas(x1,x2,x3,x4,x5)
    (w0,w1,w2)=l_weights(beta_0,beta_1,beta_2)
    
    p0=l_p0(x1,x2,x3,x4,x5)
    p1=l_p1(x1,x2,x3,x4,x5)
    p2=l_p2(x1,x2,x3,x4,x5)
    
    return w0*p0+w1*p1+w2*p2

# left WENO approximation
def dsol_wenol(sol,dx):
    def fhat(i):
        fi=sol[i-3:i+2]
        return l_weno(*fi)
    
    fhat=np.array([fhat(i) for i in range(3,len(sol)-1)])
    return (fhat[1:]-fhat[:-1])/dx


# right polynomials
def r_p0(x1,x2,x3,x4,x5):
    return (-x1+5*x2+2*x3)/6
def r_p1(x1,x2,x3,x4,x5):
    return (2*x2+5*x3-x4)/6
def r_p2(x1,x2,x3,x4,x5):
    return (11*x3-7*x4+2*x5)/6

# right nonlinear weights
def r_weights(beta_0,beta_1,beta_2):
    d0=3/10
    d1=6/10
    d2=1/10
    eps=1e-6
    alpha_0=d0/((eps+beta_0)**2)
    alpha_1=d1/((eps+beta_1)**2)
    alpha_2=d2/((eps+beta_2)**2)
    alpha_sum=alpha_0+alpha_1+alpha_2
    w0=alpha_0/alpha_sum
    w1=alpha_1/alpha_sum
    w2=alpha_2/alpha_sum
    return (w0,w1,w2)

# right numerical flux
def r_weno(x1,x2,x3,x4,x5):
    (beta_0,beta_1,beta_2)=_betas(x1,x2,x3,x4,x5)
    (w0,w1,w2)=r_weights(beta_0,beta_1,beta_2)
    
    p0=r_p0(x1,x2,x3,x4,x5)
    p1=r_p1(x1,x2,x3,x4,x5)
    p2=r_p2(x1,x2,x3,x4,x5)
    
    return w0*p0+w1*p1+w2*p2

# right WENO approximation
def dsol_wenor(sol,dx):
    def fhat(i):
        fi=sol[i-3:i+2]
        return r_weno(*fi)
    
    fhat=np.array([fhat(i) for i in range(3,len(sol)-1)])
    return (fhat[1:]-fhat[:-1])/dx





