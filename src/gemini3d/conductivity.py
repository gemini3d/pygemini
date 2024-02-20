#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:21:07 2024

@author: zettergm
"""

import numpy as np
from .phys_const import ms,qs,kB
import gemini3d.msis
import scipy as sp

def conductivity_reconstruct(time,dat,cfg,xg):
    # neutral atmospheric information
    atmos=gemini3d.msis.msis_setup(cfg,xg)
    
    # composition and extraction (or creation) of full state variables
    if "ns" in dat.keys():    # proxy for full state output file
        ns=dat["ns"]
        Ts=dat["Ts"]
        vs1=dat["vs1"]
    else:
        p=1/2+1/2*np.tanh((xg["alt"]-220e3)/20e3)
        shapevar=np.concatenate( (xg["lx"][:],[7]) )
        ns=np.zeros( shapevar )
        ns[:,:,:,0]=p*dat["ne"]
        nmolc=(1-p)*dat["ne"]
        ns[:,:,:,1]=1/3*nmolc
        ns[:,:,:,2]=1/3*nmolc
        ns[:,:,:,3]=1/3*nmolc
        ns[:,:,:,6]= dat["ne"]           #if you don't separately assign electron density the hall and parallel terms are wrong
        Ts=np.zeros( shapevar )
        for i in range(0,6):
            Ts[:,:,:,i]=dat["Ti"]
        Ts[:,:,:,6]= dat["Te"]
        vs1=np.zeros( shapevar )
        for i in range(0,7):
            vs1[:,:,:,i]=dat["v1"]
            
    # collision frequencies and conducivities
    nusn,nus,nusj,nuss,Phisj,Psisj = collisions3D(atmos,Ts,ns,vs1,ms)
    muP,muH,mu0,sigP,sigH,sig0,incap = conductivities3D(nus,nusj,ns,ms,qs,xg["Bmag"])
    
    # Integrate wrt field-line coordinate in physical units (meters), taking 
    #   care to remove one of then end ghost cells and leave one so that the 
    #   cumulative sum is conformable with first axis of simulation output data
    h1=xg["h1"][3:-1,3:-1,3:-1]
    dx1=xg["dx1b"][2:-1]
    SigP = np.zeros( xg["lx"][1:3] )
    SigH = np.zeros( xg["lx"][1:3] )
    Incap = np.zeros( xg["lx"][1:3] )
    for i2 in range(0,xg["lx"][1]):
        for i3 in range(0,xg["lx"][2]):
            dl1=h1[:,i2,i3]*dx1
            l1=np.cumsum(dl1)
            SigP[i2,i3]=np.trapz(sigP[:,i2,i3],l1)
            SigH[i2,i3]=np.trapz(sigH[:,i2,i3],l1)
            Incap[i2,i3]=np.trapz(incap[:,i2,i3],l1)
    
    return sigP,sigH,sig0,SigP,SigH,incap,Incap


##############################################################################
def collisions3D(atmos,Ts,ns,vsx1,ms):
    # convenience variables
    nO=atmos["nO"]
    nN2=atmos["nN2"]
    nO2=atmos["nO2"]
    Tn=atmos["Tn"]
    nH=atmos["nH"]
    lx1,lx2,lx3,lsp = ns.shape
    ln=4      # number of neutral species

    # output arrays
    nusn=np.zeros( (lx1,lx2,lx3,lsp,ln) )
    nus=np.zeros( (lx1,lx2,lx3,lsp) )
    nusj=np.zeros( (lx1,lx2,lx3,lsp,lsp) )
    nuss=np.zeros( (lx1,lx2,lx3,lsp) )
    Psisj=np.ones( (lx1,lx2,lx3,lsp,lsp) )
    Phisj=np.ones( (lx1,lx2,lx3,lsp,lsp) )

    # Collision frequencies of O+, NO+, N2+, O2+ with O, N, and O2.
    # Species O+
    T=0.5*(Tn+Ts[:,:,:,0]);
    ROp = 3.67e-11*(1-0.064*np.log10(T))**2*(T**0.5)*nO
    ROpH = 4.63e-12*(Tn+Ts[:,:,:,0]/16)**0.5*nH
    nusn[:,:,:,0,0] = ROp*1e-6
    nusn[:,:,:,0,1] = 6.82e-10*nN2*1e-6
    nusn[:,:,:,0,2] = 6.64e-10*nO2*1e-6
    nusn[:,:,:,0,3] = ROpH*1e-6
    nus[:,:,:,0] = np.sum(nusn[:,:,:,0,:],axis=3)

    # Species NO+
    nusn[:,:,:,1,0] = 2.44e-10*nO*1e-6
    nusn[:,:,:,1,1] = 4.34e-10*nN2*1e-6
    nusn[:,:,:,1,2] = 4.27e-10*nO2*1e-6
    nusn[:,:,:,1,3] = 0.69e-10*nH*1e-6
    nus[:,:,:,1] = np.sum(nusn[:,:,:,1,:],axis=3)

    # Species N2+
    T=0.5*(Tn+Ts[:,:,:,2])
    nusn[:,:,:,2,0] = 2.58e-10*nO*1e-6
    RN2 = 5.14e-11*(1-0.069*np.log10(T))**2*(T**0.5)*nN2
    nusn[:,:,:,2,1] =  RN2*1e-6
    nusn[:,:,:,2,2] = 4.49e-10*nO2*1e-6
    nusn[:,:,:,2,3] = 0.74e-10*nH*1e-6
    nus[:,:,:,2] = np.sum(nusn[:,:,:,2,:],axis=3)

    # Species O2+
    T=0.5*(Tn+Ts[:,:,:,3])
    nusn[:,:,:,3,0] = 2.31e-10*nO*1e-6
    nusn[:,:,:,3,1] = 4.13e-10*nN2*1e-6
    RO2 = 2.59e-11*(1-.073*np.log10(T))**2*(T**0.5)*nO2
    nusn[:,:,:,3,2] =  RO2*1e-6
    nusn[:,:,:,3,3] = 0.65e-10*nH*1e-6
    nus[:,:,:,3] = np.sum(nusn[:,:,:,3,:],axis=3)

    # Species N+
    nusn[:,:,:,4,0] = 4.42e-10*nO*1e-6
    nusn[:,:,:,4,1] = 7.47e-10*nN2*1e-6
    nusn[:,:,:,4,2] =  7.25e-10*nO2*1e-6
    nusn[:,:,:,4,3] = 1.45e-10*nH*1e-6
    nus[:,:,:,4] = np.sum(nusn[:,:,:,4,:],axis=3)

    # Species H+
    T=0.5*(Tn+Ts[:,:,:,5])
    RHpO = 6.61e-11*(1-.047*np.log10(Ts[:,:,:,5]))**2*(Ts[:,:,:,5]**0.5)*nO
    RHpH = 2.65e-10*(1-0.083*np.log10(T))**2*(T**0.5)*nH
    nusn[:,:,:,5,0] = RHpO*1e-6
    nusn[:,:,:,5,1] = 33.6e-10*nN2*1e-6
    nusn[:,:,:,5,2] = 32.0e-10*nO2*1e-6
    nusn[:,:,:,5,3] = RHpH*1e-6
    nus[:,:,:,5] = np.sum(nusn[:,:,:,5,:],axis=3)

    # Species electrons
    #    T=0.5*(Tn+Ts(:,:,:,6));
    T=Ts[:,:,:,lsp-1]
    nusn[:,:,:,lsp-1,0] = 8.9e-11*(1+5.7e-4*T)*(T**0.5)*nO*1e-6
    nusn[:,:,:,lsp-1,1] = 2.33e-11*(1-1.21e-4*T)*(T)*nN2*1e-6
    nusn[:,:,:,lsp-1,2] = 1.82e-10*(1+3.6e-2*(T**0.5))*(T**0.5)*nO2*1e-6
    nusn[:,:,:,lsp-1,3] = 4.5e-9*(1-1.35e-4*T)*(T**0.5)*nH*1e-6
    nus[:,:,:,lsp-1] = np.sum(nusn[:,:,:,lsp-1,:],axis=3)

    # Coulomb collisions
    Csj=np.array([ [0.22, 0.26, 0.25, 0.26, 0.22, 0.077, 1.87e-3],
         [0.14, 0.16, 0.16, 0.17, 0.13, 0.042, 9.97e-4],
         [0.15, 0.17, 0.17, 0.18, 0.14, 0.045, 1.07e-3],
         [0.13, 0.16, 0.15, 0.16, 0.12, 0.039, 9.347e-4],
         [0.25, 0.28, 0.28, 0.28, 0.24, 0.088, 2.136e-3],
         [1.23, 1.25, 1.25, 1.25, 1.23, 0.90,  29.7e-3],
         [54.5, 54.5, 54.5, 54.5, 54.5, 54.5,  54.5/np.sqrt(2)] ] )
    for is1 in range(0,lsp):
        for is2 in range(0,lsp):
            T=(ms[is2]*Ts[:,:,:,is1]+ms[is1]*Ts[:,:,:,is2])/(ms[is2]+ms[is1])
            nusj[:,:,:,is1,is2]=Csj[is1,is2]*ns[:,:,:,is2]*1e-6/T**1.5    # standard collision frequencies

            # FIXME: numberical issues here; I'll figure it out later...
            # mred=ms[is1]*ms[is2]/(ms[is1]+ms[is2])                     # high speed correction factors (S&N 2000)
            # Wst=np.abs(vsx1[:,:,:,is1]-vsx1[:,:,:,is2])/np.sqrt(2*kB*T/mred)
            # #Wst=max(Wst,0.01);       %major numerical issues with asymptotic form (as Wst -> 0)!!!
            # Psisj[:,:,:,is1,is2]=np.exp(-Wst**2)

            # Phinow=3/4*np.sqrt(np.pi)*sp.special.erf(Wst)/Wst**3-3/2/Wst**2*Psisj[:,:,:,is1,is2];
            # Phinow[Wst < 0.1]=1
            # Phisj[:,:,:,is1,is2]=Phinow

    for is1 in range(0,lsp):
        nuss[:,:,:,is1]=nusj[:,:,:,is1,is1]
        nusj[:,:,:,is1,is1]=np.zeros( (lx1,lx2,lx3) )     #self collisions in Coulomb array need to be zero for later code in momentum and energy sources

    return nusn,nus,nusj,nuss,Phisj,Psisj
##############################################################################


##############################################################################
def conductivities3D(nus,nusj,ns,ms,qs,B):   
    lx1,lx2,lx3,lsp=nus.shape
    mu0=np.zeros( (lx1,lx2,lx3,lsp) )
    muP=np.zeros( (lx1,lx2,lx3,lsp) )
    muH=np.zeros( (lx1,lx2,lx3,lsp) )
    mubase=np.zeros( (lx1,lx2,lx3,lsp) )
    cfact=np.zeros( (lx1,lx2,lx3,lsp) )

    # Mobilities
    for isp in range(0,lsp):
       OMs=qs[isp]*B/ms[isp]                                      # cyclotron
          
       if (not isp==lsp-1): 
           mu0[:,:,:,isp]=qs[isp]/ms[isp]/nus[:,:,:,isp]        # parallel mobility
           mubase[:,:,:,isp]=mu0[:,:,:,isp]
       else:
           nuse=np.sum(nusj[:,:,:,lsp-1,:],axis=3);
           mu0[:,:,:,lsp-1]=qs[lsp-1]/ms[lsp-1]/(nus[:,:,:,lsp-1]+nuse)
           mubase[:,:,:,lsp-1]=qs[lsp-1]/ms[lsp-1]/nus[:,:,:,lsp-1]
    
       muP[:,:,:,isp]=mubase[:,:,:,isp]*nus[:,:,:,isp]**2/(nus[:,:,:,isp]**2+OMs**2)      #Pederson
       muH[:,:,:,isp]=-1*mubase[:,:,:,isp]*nus[:,:,:,isp]*OMs/(nus[:,:,:,isp]**2+OMs**2)  #Hall    
    
    # Conductivities
    sig0=ns[:,:,:,lsp-1]*qs[lsp-1]*mu0[:,:,:,lsp-1]     #parallel includes only electrons...
    
    for isp in range(0,lsp):
       cfact[:,:,:,isp]=ns[:,:,:,isp]*qs[isp]
    sigP=np.sum(cfact*muP,axis=3)
    sigH=np.sum(cfact*muH,axis=3)
        
    # Inertial capacitance
    incap=np.zeros( (lx1,lx2,lx3) ) 
    for isp in range(0,lsp):
      incap=incap+ns[:,:,:,isp]*ms[isp]
    incap = incap/B**2

    return muP,muH,mu0,sigP,sigH,sig0,incap