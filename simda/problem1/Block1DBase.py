#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rubiop
"""
import matplotlib.pyplot as plt
import numpy as np

class BlockType:
   def __init__(self,rand_in,type_name):
      self.type_name=type_name
      self.rand_hib=rand_in[0]
      self.rand_hs=rand_in[1]
      self.rand_hit=rand_in[2]
      self.rand_id=rand_in[3]
      self.rand_sd=rand_in[4]
      self.rand_hi=rand_in[5]
      self.rand_l=rand_in[6]
   def rand_v(self):
       return np.array([self.rand_hib,self.rand_hs,self.rand_hit,self.rand_id,self.rand_sd,self.rand_hi,self.rand_l])
   def randomize(self,rand_in):
      self.rand_hib=float(rand_in[0])
      self.rand_hs=float(rand_in[1])
      self.rand_hit=float(rand_in[2])
      self.rand_id=float(rand_in[3])
      self.rand_sd=float(rand_in[4])
      self.rand_hi=float(rand_in[5])
      self.rand_l=float(rand_in[6])
   def f_hs(self,x):
      raise NotImplementedError('Not implemented for this block type')
   def f_hib(self,x):
      raise NotImplementedError('Not implemented for this block type')
   def f_hit(self,x):
      raise NotImplementedError('Not implemented for this block type')
   def l(self):
      raise NotImplementedError('Not implemented for this block type')
   def hi_o(self):
      raise NotImplementedError('Not implemented for this block type')
   def rho_i(self):
      raise NotImplementedError('Not implemented for this block type')
   def rho_s(self):
      raise NotImplementedError('Not implemented for this block type')

class Block1D: #reference block of 
    def __init__(self,block_type,x_o=np.array([0])):
        self.l=block_type.l() #block length
        self.hi_o=block_type.hi_o() #ice thickness at x_0
        self.x_o=x_o #position of the block reference
        self.f_hs=block_type.f_hs  #geometry of the ice-snow interface (ice roughness)
        self.f_hib=block_type.f_hib  #geometry of the water-ice interface
        self.f_hit=block_type.f_hit  #snow accumulation function (global function)
        self.rho_i=block_type.rho_i() #ice density
        self.rho_s=block_type.rho_s() #snow density
        self.type_name=block_type.type_name
        #print('f_hs', self.f_hs(np.array([0])))
    def snow_depth(self,x, *args, **kwargs): 
        #f must be a one dim. function
        #print('f_hs', self.f_hs(np.array([0])))
        idx_in_domain=(x <= self.x_o+self.l) * (x >= self.x_o)
        x_in_domain=x[idx_in_domain]
        hs=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            hs[idx_in_domain]=self.f_hs(x_in_domain)
        return hs
    def bot_ice_roughness(self,x,*args, **kwargs):#geometry of the water-ice interface
        #f must be a one dim. function
        idx_in_domain=(x <= self.x_o+self.l) * (x >= self.x_o)
        x_in_domain=x[idx_in_domain]
        hr_bot=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            hr_bot[idx_in_domain]=self.f_hib(x_in_domain-self.x_o)
            #hr_bot[hr_bot>self.hi_o]=self.hi_o-0.04
        return hr_bot
    def top_ice_roughness(self,x,*args, **kwargs): #geometry of the ice-snow interface
        #f must be a one dim. function
        
        idx_in_domain=(x <= self.x_o+self.l) * (x >= self.x_o)
        x_in_domain=x[idx_in_domain]
        hr_top=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            hr_top[idx_in_domain]=self.f_hit(x_in_domain-self.x_o)
            #hr_top[hr_top<-self.hi_o]=-self.hi_o+0.04
        return hr_top
    def ice_thickness(self,x,*args, **kwargs):
        idx_in_domain=(x <= self.x_o+self.l) * (x >= self.x_o)
        x_in_domain=x[idx_in_domain]
        hi=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            hi[idx_in_domain]=self.hi_o+self.bot_ice_roughness(x_in_domain)+self.top_ice_roughness(x_in_domain)
            #hi[hi<0]=0.02     
        return hi
    def freeboard(self,x,rho_w=1024,*args, **kwargs): #height of ice above sea level (removing the snow!!) (more like mean height of top ice above sea)
        #f must be a one dim. function 
        idx_in_domain=((x <= self.x_o+self.l) * (x >= self.x_o)).flatten()
        x_in_domain=x[idx_in_domain]
        hf=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            x_block=np.linspace(self.x_o,self.x_o+self.l,200)
            snow_depth_mean=np.mean(self.snow_depth(x_block))
            ice_thickness_mean=np.mean(self.ice_thickness(x_block))
            freeboard_mean=(ice_thickness_mean*(rho_w-self.rho_i)-snow_depth_mean*(self.rho_s-rho_w))/rho_w
            freeboard_mean=freeboard_mean-snow_depth_mean
            hf[idx_in_domain]=freeboard_mean*np.ones(x_in_domain.shape)
        return hf
    def ice_depth(self,x,*args, **kwargs): #equation of bottom ice interface (w.r.t sea level)
        idx_in_domain=((x <= self.x_o+self.l) * (x >= self.x_o)).flatten()
        x_in_domain=x[idx_in_domain]
        z_wi=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:       
            z_wi[idx_in_domain]=-(self.hi_o+self.bot_ice_roughness(x_in_domain))+self.freeboard(x_in_domain)
        return z_wi
    def ice_snow_interface(self,x,*args, **kwargs): #equation of top ice interface (w.r.t sea level)
        idx_in_domain=((x <= self.x_o+self.l) * (x >= self.x_o)).flatten()
        x_in_domain=x[idx_in_domain]
        z_is=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size: 
            z_is[idx_in_domain]=self.freeboard(x_in_domain)+self.top_ice_roughness(x_in_domain)
        return z_is
    def snow_air_interface(self,x,*args, **kwargs): #equation of sea ice surface (w.r.t sea level)
        idx_in_domain=((x <= self.x_o+self.l) * (x >= self.x_o)).flatten()
        x_in_domain=x[idx_in_domain]
        z_sa=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:
            snow_depth_pos=self.snow_depth(x_in_domain).copy() #in case snow function removes snow!
            snow_depth_pos[snow_depth_pos<0]=0  
            z_sa[idx_in_domain]=self.freeboard(x_in_domain)+self.top_ice_roughness(x_in_domain)+snow_depth_pos
        return z_sa
    def intensity(self,x,*args, **kwargs): #intensity of the ice top
        #Should be also a function of snow depth?
        idx_in_domain=((x <= self.x_o+self.l) * (x >= self.x_o)).flatten()
        x_in_domain=x[idx_in_domain]
        i_t=np.zeros(x.shape)
        if self.type_name != 'Water' and x_in_domain.size:      
            i_t[idx_in_domain]=1e-4/(self.ice_thickness(x_in_domain)+1e-4)
        return i_t
    def plot(self,x,ax,*args, **kwargs): #plot function!
        x_p=x[(x <= self.x_o+self.l) * (x >= self.x_o)]

        x_s=np.concatenate((self.x_o,self.x_o))
        x_e=np.concatenate((self.x_o+self.l,self.x_o+self.l))

        ax.plot(x_p,self.snow_air_interface(x_p),'k',linewidth=0.25)
        ax.plot(x_p,np.zeros(x_p.shape),'-k',linewidth=0.5)
        ax.plot(x_p,self.ice_depth(x_p),'-k',linewidth=0.25)
        ax.plot(x_p,self.ice_snow_interface(x_p),'-k',linewidth=0.25)
        
        if x_s[0] <= x[-1] and x_s[0] >= x[0]:
             ax.plot(x_s,[self.ice_depth(x_s[:1]),self.snow_air_interface(x_s[:1])],'k',linewidth=0.25)
        
        if x_e[0] <= x[-1] and x_e[0] >= x[0]:
            ax.plot(x_e,[self.ice_depth(x_e[:1]),self.snow_air_interface(x_e[:1])],'k',linewidth=0.25)
        
        ax.fill_between(x_p,self.ice_depth(x_p),self.ice_snow_interface(x_p),facecolor=['#94C4F5'])
        ax.fill_between(x_p,self.snow_air_interface(x_p),self.ice_snow_interface(x_p),facecolor=['#DDE0E4'])


