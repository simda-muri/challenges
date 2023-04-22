#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rubiop
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from Block1DBase import BlockType, Block1D

from scipy.interpolate import interp1d


m=20

kernel = 0.02 * RBF(length_scale=0.1)
gpr= GaussianProcessRegressor(kernel=kernel,copy_X_train=True,random_state=int(1e9))
x_ref=np.linspace(0,7,140) 
X = x_ref.reshape(-1, 1)
y_sample = gpr.sample_y(X, 4000,random_state=int(1e9))
y_mean, y_std = gpr.predict(X, return_std=True)
f_hit_myi=y_sample+2*y_std[0]
u, s, vh = np.linalg.svd(f_hit_myi)
vh=vh.T
u_hit_myi=u*s
u_hit_myi=u_hit_myi[:,:m]
vh=vh[:,:m]
m_hit_myi=np.mean(vh,axis=0)
cov_hit_myi=np.cov(vh.T)


kernel = 0.4 * RBF(length_scale=0.65)
gpr= GaussianProcessRegressor(kernel=kernel,copy_X_train=True,random_state=int(1e9))
x_ref=np.linspace(0,7,70) 
X = x_ref.reshape(-1, 1)
y_sample = gpr.sample_y(X, 4000,random_state=int(1e9))
y_mean, y_std = gpr.predict(X, return_std=True)
f_hib_myi=y_sample+2*y_std[0]

u, s, vh = np.linalg.svd(f_hib_myi)
vh=vh.T
u_hib_myi=u*s
u_hib_myi=u_hib_myi[:,:m]
vh=vh[:,:m]
m_hib_myi=np.mean(vh,axis=0)
cov_hib_myi=np.cov(vh.T)


kernel= 0.002 * RBF(length_scale=0.5)
gpr= GaussianProcessRegressor(kernel=kernel,copy_X_train=True,random_state=int(1e9))
x_ref=np.linspace(0,7,70) 
X = x_ref.reshape(-1, 1)
y_sample = gpr.sample_y(X, 4000,random_state=int(1e9))
y_mean, y_std = gpr.predict(X, return_std=True)
f_hit_fyi=y_sample+2*y_std[0]

u, s, vh = np.linalg.svd(f_hit_fyi)
vh=vh.T
u_hit_fyi=u*s
u_hit_fyi=u_hit_fyi[:,:m]
vh=vh[:,:m]
m_hit_fyi=np.mean(vh,axis=0)
cov_hit_fyi=np.cov(vh.T)


kernel = 0.02 * RBF(length_scale=1.7)
gpr= GaussianProcessRegressor(kernel=kernel,copy_X_train=True,random_state=int(1e9))
x_ref=np.linspace(0,7,70) 
X = x_ref.reshape(-1, 1)
y_sample = gpr.sample_y(X, 4000,random_state=int(1e9))
y_mean, y_std = gpr.predict(X, return_std=True)
f_hib_fyi=y_sample+2*y_std[0]

u, s, vh = np.linalg.svd(f_hib_fyi)
vh=vh.T
u_hib_fyi=u*s
u_hib_fyi=u_hib_fyi[:,:m]
vh=vh[:,:m]
m_hib_fyi=np.mean(vh,axis=0)
cov_hib_fyi=np.cov(vh.T)

def GP_fixed(mu,cov,u,seed):
   rng = np.random.RandomState(seed)
   coeff=rng.multivariate_normal(mu,cov)
   y_new=np.zeros(u[:,0].shape)
   for k in range(20):
      y_new=y_new+coeff[k]*u[:,k]
   return y_new


class FYIceBlock_real(BlockType):
   def __init__(self):
      super(FYIceBlock_real,self).__init__(np.random.rand(7,),'FYI')
      self.f_hit_fixed=GP_fixed(m_hit_fyi,cov_hit_fyi,u_hit_fyi,int(1e9*self.rand_hit))
      self.f_hib_fixed=GP_fixed(m_hib_fyi,cov_hib_fyi,u_hib_fyi,int(1e9*self.rand_hib))
   def f_hs(self,x):
      e_s=0.1*self.rand_hs
      f_hs = e_s*np.ones(x.shape)
      return f_hs.flatten()
   def f_hib(self,x):
      x_ref=np.linspace(0,7,70) 
      f_hib=interp1d(x_ref.flatten(), self.f_hib_fixed.flatten(),kind='linear',fill_value=0)
      return f_hib(x).flatten()
   def f_hit(self,x):
      x_ref=np.linspace(0,7,70) 
      f_hit=interp1d(x_ref.flatten(), self.f_hit_fixed.flatten(),kind='linear',fill_value=0)
      return f_hit(x).flatten()
   def l(self):
      return np.array([4+0.75*self.rand_l])
   def hi_o(self):
      return np.array([0.20+1*self.rand_hi])
   def rho_i(self):
      return 916.7-35.7+2*35.7*self.rand_id
   def rho_s(self):
      return 324-50+100*self.rand_sd

class MYIceBlock_real(BlockType):
   def __init__(self):
      super(MYIceBlock_real,self).__init__(np.random.rand(7,),'MYI')
      self.f_hit_fixed=GP_fixed(m_hit_myi,cov_hit_myi,u_hit_myi,int(1e9*self.rand_hit))
      self.f_hib_fixed=GP_fixed(m_hib_myi,cov_hib_myi,u_hib_myi,int(1e9*self.rand_hib))
   def f_hs(self,x):
      e_s=0.35-0.06+0.12*self.rand_hs
      f_hs = e_s*np.ones(x.shape)
      return f_hs.flatten()
   def f_hib(self,x):
      # kernel = 0.4 * RBF(length_scale=0.65)
      # gpr= GaussianProcessRegressor(kernel=kernel, random_state=int(1e9*self.rand_hib))
      # x_ref=np.linspace(0,6,60) 
      # X = x_ref.reshape(-1, 1)
      # y_sample = gpr.sample_y(X, 1,random_state=int(1e9*self.rand_hit))
      # y_mean, y_std = gpr.predict(X, return_std=True)
      # f_hib_r=y_sample+2*y_std[0]
      # f_hib=interp1d(x_ref.flatten(), f_hib_r.flatten(),kind='linear',fill_value=0)
      x_ref=np.linspace(0,7,70) 
      f_hib=interp1d(x_ref.flatten(), self.f_hib_fixed.flatten(),kind='linear',fill_value=0)
      return f_hib(x).flatten()
   def f_hit(self,x):
      # kernel = 0.02 * RBF(length_scale=0.1)
      # gpr= GaussianProcessRegressor(kernel=kernel,copy_X_train=True,random_state=int(1e9*self.rand_hit))
      # x_ref=np.linspace(0,6,120) 
      # X = x_ref.reshape(-1, 1)
      # y_sample = gpr.sample_y(X, 1,random_state=int(1e9*self.rand_hit))
      # y_mean, y_std = gpr.predict(X, return_std=True)
      # f_hit_r=y_sample+2*y_std[0]
      # f_hit=interp1d(x_ref.flatten(), f_hit_r.flatten(),kind='linear',fill_value=0)
      x_ref=np.linspace(0,7,140) 
      f_hit=interp1d(x_ref.flatten(), self.f_hit_fixed.flatten(),kind='linear',fill_value=0)
      return f_hit(x).flatten()
   def l(self):
      return np.array([3+0.75*self.rand_l])
   def hi_o(self):
      return np.array([1.5+3*self.rand_hi])
   def rho_i(self):
      return 882-23+46*self.rand_id
   def rho_s(self):
      return 320-20+40*self.rand_sd

class WaterBlock_real(BlockType):
   def __init__(self):
      super(WaterBlock_real,self).__init__(np.random.rand(7,),'Water')
   def f_hs(self,x):
      f_hs = np.zeros(x.shape)
      return f_hs.flatten()
   def f_hib(self,x):
      f_hib = np.zeros(x.shape)
      return f_hib.flatten()
   def f_hit(self,x):
      f_hit = np.zeros(x.shape)
      return f_hit.flatten()
   def l(self):
      return np.array([2.5+1*self.rand_l])
   def hi_o(self):
      return np.array([0])
   def rho_i(self):
      return 1024
   def rho_s(self):
      return 324


class IceRegion:
    def __init__(self,Nb,fyi_block,myi_block,water_block,r_fyi,r_myi): #Could be generalize to N ice types
        if r_fyi+r_myi<=0 or r_fyi+r_myi>1:
            raise ValueError('Sum of ratios of MYI and FYI should be between [0,1]')
        if r_fyi<0 or r_fyi>1:
            raise ValueError('Ratio of FYI should be in [0,1]')
        if r_myi<0 or r_myi>1:
            raise ValueError('Ratio of MYI should be in [0,1]')
        self.Nb=Nb
        self.fyi_block=fyi_block
        self.myi_block=myi_block
        self.water_block=water_block
        self.r_fyi=r_fyi
        self.r_myi=r_myi
        L_type=np.zeros((Nb,))
        L_type[0:int(Nb*r_fyi)]=1
        L_type[int(Nb*r_fyi):int(Nb*r_fyi)+int(Nb*r_myi)]=2
        L_type=np.random.permutation(L_type)

        x_o=np.array([0])
        x_b=x_o
        L_blocks=[]
        L_name_type=[]
        for k,b_typ in enumerate(L_type):
            myi_block.randomize(np.random.rand(7,))
            fyi_block.randomize(np.random.rand(7,))
            water_block.randomize(np.random.rand(7,))
            if b_typ==1:
                block=Block1D(FYIceBlock_real(),x_o)
                L_name_type.append(block.type_name)
            elif b_typ==2:
                block=Block1D(MYIceBlock_real(),x_o)
                L_name_type.append(block.type_name)
            else: #Water
                block=Block1D(WaterBlock_real(),x_o)
                L_name_type.append(block.type_name)
            x_o=block.x_o+block.l
            x_e=x_o
            L_blocks.append(block)
        self.x_e=x_e
        self.x_b=x_b
        self.block_list=L_blocks
        self.type_names=L_name_type

    def set_snow_depth(self,f_hs):
        for block_e in self.block_list:
            block_e.f_hs=f_hs
        return 'Global snow accumalation has been set'
    
    def elevation(self,x):
        elev=np.zeros(x.shape)
        for block_e in self.block_list:
            elev=elev+block_e.snow_air_interface(x)
        return elev

    def top_ice_roughness(self,x):
        top_ir=np.zeros(x.shape)
        for block_e in self.block_list:
            top_ir=top_ir+block_e.top_ice_roughness(x)
        return top_ir

    def top_ice_elevation(self,x):
        top_ir=np.zeros(x.shape)
        for block_e in self.block_list:
            top_ir=top_ir+block_e.ice_snow_interface(x)
        return top_ir
   
    def ice_depth(self,x,*args, **kwargs): #equation of bottom ice interface (w.r.t sea level)
        bot_ir=np.zeros(x.shape)
        for block_e in self.block_list:
            bot_ir=bot_ir+block_e.ice_depth(x)
        return bot_ir

    def intensity(self,x):
        int=np.zeros(x.shape)
        for block_e in self.block_list:
            int=int+block_e.intensity(x)
        return int

    def ice_thickness(self,x):
        ice_thick=np.zeros(x.shape)
        for block_e in self.block_list:
            ice_thick=ice_thick+block_e.ice_thickness(x)
        return ice_thick

    def plot_region(self,x,ax):
        ax.fill_between(x,np.zeros(x.shape),-6.5*np.ones(x.shape),facecolor=['#146E9B'])
        for block_e in self.block_list:
           block_e.plot(x,ax)
