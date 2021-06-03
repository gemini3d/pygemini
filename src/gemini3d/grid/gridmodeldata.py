#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:28:13 2021

Various transformations needed to grid model output so it can be easily plotted

@author: zettergm
"""

# imports
import numpy as np
from numpy import pi
import sys
import scipy.interpolate
from .convert import Re

#Grid the scalar GEMINI output data in parm onto a regular *geomagnetic* coordinates
#grid.  By default create a linearly spaced output grid based on
#user-provided limits (or grid limits).  Needs to be updated to deal with
#2D input grids; can interpolate from 3D grids to 2D slices...
#
#  [alt,mlon,mlat,parmi]=model2magcoords(xg,parm,lalt,llon,llat,altlims,mlonlims,mlatlims)
def model2magcoords(xg,parm,lalt,llon,llat,altlims=None,mlonlims=None,mlatlims=None):    
    # convenience variables
    mlon=xg["phi"]*180/pi
    mlat=90-xg["theta"]*180/pi
    alt=xg["alt"]
    lx1=xg["lx"][0]; lx2=xg["lx"][1]; lx3=xg["lx"][2];
    inds1=range(2,lx1+2); inds2=range(2,lx2+2); inds3=range(2,lx3+2);
    x1=xg["x1"][inds1]; x2=xg["x2"][inds2]; x3=xg["x3"][inds3];
    
    # set some defaults if not provided by user
    if altlims==None:
        altlims=np.array( [np.min(alt.flatten())+0.0001,np.max(alt.flatten())-0.0001] )
        mlonlims=np.array( [np.min(mlon.flatten())+0.0001,np.max(mlon.flatten())-0.0001] )
        mlatlims=np.array( [np.min(mlat.flatten())+0.0001,np.max(mlat.flatten())-0.0001] )

    # define uniform grid in magnetic coords.
    alti=np.linspace(altlims[0],altlims[1],lalt)
    mloni=np.linspace(mlonlims[0],mlonlims[1],llon)
    mlati=np.linspace(mlatlims[0],mlatlims[1],llat)
    [ALTi,MLONi,MLATi]=np.meshgrid(alti,mloni,mlati,indexing="ij")

    # identify the type of grid that we are using
    minh1=np.min(xg["h1"])
    maxh1=np.max(xg["h1"])
    if (abs(minh1-1)>1e-4 or abs(maxh1-1)>1e-4):    # curvilinear, dipole
        flagcurv=1
    else:                                            # cartesian
        flagcurv=0
        # elif others possible...

    # Compute the coordinates of the intended interpolation grid IN THE MODEL SYSTEM/BASIS.
    # There needs to be a separate transformation here for each coordinate system that the model
    # may use...
    if (flagcurv==1):
        [qi,pei,phii]=geomag2dipole(ALTi,MLONi,MLATi)
        x1i=qi; x2i=pei; x3i=phii;
    elif (flagcurv==0):
        [zUENi,xUENi,yUENi]=geomag2UENgeomag(ALTi,MLONi,MLATi)
        x1i=zUENi; x2i=xUENi; x3i=yUENi;
    else:
        sys.error('Unsupported grid type...')

    # Execute plaid interpolation
    #[X1,X2,X3]=np.meshgrid(x1,x2,x3,indexing="ij")
    xi=np.zeros((x1i.size,3))
    xi=np.array( (x1i.flatten(),x2i.flatten(),x3i.flatten()) ).transpose()
    parmi=scipy.interpolate.interpn( (x1,x2,x3), np.array(parm), xi, method="linear", bounds_error=False, fill_value=np.NaN)    
    parmi=np.reshape(parmi,[lalt,llon,llat])

    return [alti,mloni,mlati,parmi]


# Convert geomagnetic coordinates into dipole
def geomag2dipole(alt,mlon,mlat):
    theta=pi/2-mlat*pi/180
    phi=mlon*pi/180
    r=alt+Re
    q=((Re/r)**2)*np.cos(theta)
    p=r/(Re*np.sin(theta)**2)
    return [q,p,phi]


# Convert geomagnetic to UEN geomagnetic coords.
def geomag2UENgeomag(alt,mlon,mlat):
    theta=pi/2-mlat*pi/180
    phi=mlon*pi/180
    meantheta=np.mean(theta.flatten())
    meanphi=np.mean(phi.flatten())
    yUEN=-1*Re*(theta-meantheta)                   # north dist. runs backward from zenith angle
    xUEN=Re*np.sin(meantheta)*(phi-meanphi)        # some warping done here (using meantheta)
    zUEN=alt
    return [zUEN,xUEN,yUEN]