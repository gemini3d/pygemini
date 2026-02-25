#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:40:15 2023

@author: redden
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:49:51 2022

@author: mer
"""

import gemini3d.read
import gemini3d.msis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os as os
import xarray as xr

from .collisions3D import collisionfrequency

def conductivity(
    path: Path,
    time: datetime | None = None,
) -> None:

    """
    Parameters
    ----------

    path: pathlib.Path
        filename or directory + time to plot
    time: datetime.datetime, optional
        if path is a directory, time is required
    """



    kb = 1.38e-23
    amu = 1.66e-27
    e = 1.60e-19
    B = 5.0e-5 #Teslas
    m_e = 9.11e-31
    m_Oplus = 2.21e-27
    m_NOplus = 4.15e-27
    m_N2plus = 3.87e-27
    m_O2plus = 4.43e-27
    m_Nplus = 1.94e-27
    m_Hplus = 1.38e-28
    mass_ions = [m_Oplus, m_NOplus, m_N2plus, m_O2plus, m_Nplus, m_Hplus]
    mass_ions_total = m_Oplus + m_NOplus + m_N2plus + m_O2plus + m_Nplus + m_Hplus
    #mi = xr.DataArray(dims=("ion"), coords={"O+":m_Oplus, "NO+":m_NOplus, "N2+":m_N2plus, "O2+":m_O2plus, "N+":m_Nplus, "H+":m_Hplus})
    #mi = xr.DataArray(dims=("ion"), coords=('mass',["O+", "NO+", "N2+", "O2+", "N+", "H+"]), data=np.array([[m_Oplus], [m_NOplus], [m_N2plus], [m_O2plus], [m_Nplus], [m_Hplus]]))
    mi = xr.DataArray(dims=("species"), coords=[("species", ["O+", "NO+", "N2+", "O2+", "N+", "H+"])], data=[m_Oplus, m_NOplus, m_N2plus, m_O2plus, m_Nplus, m_Hplus])
    
    # O = 16, N2 = 28, O2 = 32, N = 14, NO = 30, H =1 


    # Read in GEMINI data
    dat = gemini3d.read.frame(path, time, var=["ne", "Te", "Ti", "ns", "Ts"]) 

    ns = dat["ns"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])
    ni = ns.drop("e", dim="species")
    ne = dat["ne"]
    #print(ni)
    
    # Get collision frequencies
    nu_in, nu_en, _, _ = collisionfrequency(path, time)

#    print(nu_in)
#    print(nu_en)
    
    # Calculate ion and eletron collision frequencies by summing over all neutral species
    nui = nu_in.sum(dim=("neutral"))
    nue = nu_en.sum(dim=("neutral"))

    #print(nu_i)
    #print(nu_e)


    # Ion conductivity (Schunk and Nagy equation 5.112)
    #sigma_ion = (n_ion * e_ion**2) / (mass_ion * ion-netrals_coll_freq) (Schunk and Nagy Eq 5.112) 
    sigma_i = (ni * e**2) / (mi * nui)


    # Electron conductivity
    
    #sigma_e = (ne * e**2) / (mass_e * elec_coll_freq) (Schunk and Nagy Eq 5.115)
    # elec_coll_freq is the collision frequency of the electrons summed over all
    # neutral species
    sigma_e = (ne * e**2) / (m_e * nue)

    #print('SIGMA')
    #print(sigma_i)
    #print(sigma_e)




    # Pedersen conductivity (S&N equation 5.119)
    Omgi = e * B / mi
    Omge = e * B / m_e

    sigma_P = (sigma_i * nui**2 / (nui**2 + Omgi**2)).sum(dim=("species")) + sigma_e * nue**2 / (nue**2 + Omge**2)

    #print('SIGMA P')
    #print(sigma_P)
    
    sigma_H = -(sigma_i * nui*Omgi / (nui**2 + Omgi**2)).sum(dim=("species")) + sigma_e * nue*Omge / (nue**2 + Omge**2)

    #print('SIGMA H')
    #print(sigma_H)
    
##    sigma_P_1 = np.zeros(np.shape(ne_1))
##    
##    for i in np.arange(0,6,1):
##        sigma_P_1 += (sigma_ion_1[i] * (nu_ion_1[i]**2 / (nu_ion_1[i]**2 + 
##        (e * B / mass_ions[i])**2)))
##    
##    print('sigma_P shape =', sigma_P_1.shape)
#    
#    
#    # Hall conductivity (S&N equation 5.120)
#    
#    sigma_H_ion_1 = np.zeros(np.shape(ne_1))
#    
#    for j in range(0,6,1):
#        sigma_H_ion_1 += (sigma_ion_1[j] * ((nu_ion_1[j] * ((e * B) / mass_ions[j]) / (nu_ion_1[j]**2 + 
#        (e * B / mass_ions[j])**2)))) 
#        #print('Hall conductivity =', sigma_H)
#        
#        
#    sigma_H_1 = -sigma_H_ion_1 + ((nu_en_total_1 * sigma_e_1) / ((e * B) / m_e))
#        
#    print('sigma_H shape =', sigma_H_1.shape)
#    
#
#
#
#
#    # ion-neutrals_coll_freq is the collision frequency of each individual ion summed over
#    # neutral species
#
#
#    ##direc = "/Volumes/Elements_2/Simulation_10/"
#    #direc  = "~/Desktop/2nd_Paper_Stuff/Round_grad_scale/"
#    #cfg = gemini3d.read.config(direc)
#    #cfg = gemini3d.read.config(path)
#    #print(cfg)
#    #times = cfg["time"]
#    
#    #xg = gemini3d.read.grid(direc)
#    #xg = gemini3d.read.grid(path)
#    #print(xg)
#    
#    #dat = gemini3d.read.frame(direc, datetime(2013,2,20,5,80,00), var=["ne", "Te", "Ti", "ns", "Ts"]) 
#    #dat = gemini3d.read.frame(path, time, var=["ne", "Te", "Ti", "ns", "Ts"]) 
#    print(dat.keys())
#    print(dat)
#    
#    ne_1 = np.array(dat["ne"])
#    #print('ne.shape =', ne.shape)
#    Te_1 = np.array(dat["Te"])
#    Ti_1 = np.array(dat["Ti"])
#    ns_1 = np. array(dat["ns"])
#    Ts_1 = np.array(dat["Ts"])
#    
#    z = np.array(xg["x1"][2:-2])
#    x = np.array(xg["x2"][2:-2])
#    y = np.array(xg["x3"][2:-2])
#    
#    refalt = 120e3
#    iz = np.argmin(abs(z-refalt))     # find the index where altitude is 
#            #closest to reference altitude
#    print('iz = ', iz)
#        
#    n_Oplus_1 = ns_1[0,:,:,:]
#    n_NOplus_1 = ns_1[1,:,:,:]
#    n_N2plus_1 = ns_1[2,:,:,:]
#    n_O2plus_1 = ns_1[3,:,:,:]
#    n_Nplus_1 = ns_1[4,:,:,:]
#    n_Hplus_1 = ns_1[5,:,:,:]
#    ion_dens_1 = [n_Oplus_1, n_NOplus_1, n_N2plus_1, n_O2plus_1, n_Nplus_1, n_Hplus_1]
#        
#    T_Oplus_1 = Ts_1[0,:,:,:]
#    T_NOplus_1 = Ts_1[1,:,:,:]
#    T_N2plus_1 = Ts_1[2,:,:,:]
#    T_O2plus_1 = Ts_1[3,:,:,:]
#    T_Nplus_1 = Ts_1[4,:,:,:]
#    T_Hplus_1 = Ts_1[5,:,:,:]
#    Te_1 = Ts_1[6,:,:,:]
#    ion_temp_1 = [T_Oplus_1, T_NOplus_1, T_N2plus_1, T_O2plus_1, T_Nplus_1, T_Hplus_1]
#    
#    # Ion conductivity (Schunk and Nagy equation 5.112)
#    
#    # sigma_ion = (n_ion * e_ion**2) / (mass_ion * ion-netrals_coll_freq) (Schunk and Nagy Eq 5.112) 
#    # ion-neutrals_coll_freq is the collision frequency of each individual ion summed over
#    # neutral species
#    
#    #os.environ["GEMINI_ROOT"] = "~/libgem/"
#    msisdata = gemini3d.msis.msis_setup(cfg,xg)
#    #print(msisdata.keys())
#        
#    n_O_1 = np.array(msisdata["nO"])
#    print('n_O.shape =', n_O_1.shape)
#    n_N2_1 = np.array(msisdata["nN2"])
#    n_O2_1 = np.array(msisdata["nO2"])
#    n_N_1 = np.array(msisdata["nN"])
#    n_H_1 = np.array(msisdata["nH"])
#    Tn_1 = np.array(msisdata["Tn"])
#        
#    ntrl_dens_1 = [n_O_1, n_N2_1, n_O2_1, n_N_1, n_H_1] 
#    
#    # For individual ion-neutral pairs, the non-resonant collision frequency is given by
#    # S&N equation (4.146): nu_in_nonres = C_in * n_n.  C_in values are listed in 
#    # Table 4.4.  n_n is in units of cm^-3.
#      
#    nu_in_nonres_Oplus_1 = (6.82e-10 * (n_N2_1 *1e-6) + 6.64e-10 * (n_O2_1 *1e-6) + 
#                          4.62e-10 * (n_N_1 * 1e-6))
#    
#    nu_in_nonres_NOplus_1 = (2.44e-10 * (n_O_1 * 1e-6) + 4.34e-10 * (n_N2_1 * 1e-6) + 
#                           4.27e-10 * (n_O2_1 * 1e-6) + 2.79e-10 * (n_N_1 * 1e-6) + 
#                           0.69e-10 * (n_H_1 * 1e-6))
#    
#    nu_in_nonres_N2plus_1 = (2.58e-10 * (n_O_1 * 1e-6) + 4.49e-10 * (n_O2_1 * 1e-6) + 
#                           2.95e-10 * (n_N_1 * 1e-6) + 0.74e-10 * (n_H_1 * 1e-6))
#    
#    nu_in_nonres_O2plus_1 = (2.31e-10 * (n_O_1 * 1e-6) + 4.13e-10 * (n_N2_1 * 1e-6) + 
#                           2.64e-10 * (n_N_1 * 1e-6) + 0.65e-10 * (n_H_1 * 1e-6))
#    
#    nu_in_nonres_Nplus_1 = (4.42e-10 * (n_O_1 * 1e-6) + 7.47e-10 * (n_N2_1 * 1e-6) + 
#                          7.25e-10 * (n_O2_1 * 1e-6) + 1.45e-10 * (n_H_1 * 1e-6))
#    
#    nu_in_nonres_Hplus_1 = (33.6e-10 * (n_N2_1 * 1e-6) + 32.0e-10 * (n_O2_1 * 1e-6) + 
#                          26.1e-10 * (n_N_1 * 1e-6))
#    
#    nu_in_nonres_total_1 = (nu_in_nonres_Oplus_1 + nu_in_nonres_NOplus_1 + nu_in_nonres_N2plus_1 +
#        nu_in_nonres_O2plus_1 + nu_in_nonres_Nplus_1 + nu_in_nonres_Hplus_1)
#    
#    # For individual ion-neutral pairs, the resonant collision frequencies are given by the 
#    # equations in Table 4.5
#    
#    #breakpoint()
#    nu_in_res_OplusO_1 = 3.67e-11 * (n_O_1 * 1e-6) * np.sqrt((T_Oplus_1 + Tn_1) / 2) * (1.0 - 0.064 * np.log10((T_Oplus_1 + Tn_1) / 2))**2
#    
#    nu_in_res_OplusH_1 = 4.63e-12 * (n_H_1 * 1e-6) * np.sqrt((Tn_1 + T_Oplus_1) / 16)
#    
#    nu_in_res_N2plusN2_1 = 5.14e-11 * (n_N2_1 * 1e-6) * np.sqrt((T_N2plus_1 + Tn_1) / 2) * (1.0 - 0.069 * np.log10((T_N2plus_1 + Tn_1) / 2))**2
#    
#    nu_in_res_O2plusO2_1 = 2.59e-11 * (n_O2_1 * 1e-6) * np.sqrt((T_O2plus_1 + Tn_1) / 2) * (1.0 - 0.073 * np.log10((T_O2plus_1 + Tn_1) / 2))**2
#    
#    nu_in_res_NplusN_1 = 3.83e-11 * (n_N_1 * 1e-6) * np.sqrt((T_Nplus_1 + Tn_1)/2) * (1.0 - 0.063 * np.log10((T_Nplus_1 + Tn_1) / 2))**2
#    
#    nu_in_res_HplusH_1 = 2.65e-10 * (n_H_1 * 1e-6) * np.sqrt((T_Hplus_1 + Tn_1) / 2) * (1.0 - 0.083 * np.log10((T_Hplus_1 + Tn_1) / 2))**2
#    
#    nu_in_res_HplusO_1 = 6.61e-11 * (n_O_1 * 1e-6) * np.sqrt(T_Hplus_1) * (1.0 - 0.047 * np.log10(T_Hplus_1))**2
#    
#    nu_in_res_total_1 = (nu_in_res_OplusO_1 + nu_in_res_OplusH_1 + nu_in_res_N2plusN2_1 + nu_in_res_O2plusO2_1 + 
#                       nu_in_res_NplusN_1 + nu_in_res_HplusH_1)
#    
#    nu_Oplus_1 = nu_in_nonres_Oplus_1 + nu_in_res_OplusO_1 + nu_in_res_OplusH_1
#        
#    nu_NOplus_1 = nu_in_nonres_NOplus_1
#        
#    nu_N2plus_1 = nu_in_nonres_N2plus_1 + nu_in_res_N2plusN2_1
#        
#    nu_O2plus_1 =  nu_in_nonres_O2plus_1 + nu_in_res_O2plusO2_1
#        
#    nu_Nplus_1 = nu_in_nonres_Nplus_1 + nu_in_res_NplusN_1
#        
#    nu_Hplus_1 = nu_in_nonres_Hplus_1 + nu_in_res_HplusH_1 + nu_in_res_HplusO_1
#        
#    nu_ion_1 = [nu_Oplus_1, nu_NOplus_1, nu_N2plus_1, nu_O2plus_1, nu_Nplus_1, nu_Hplus_1]
#    
#    sigma_Oplus_1 = (n_Oplus_1 * e**2) / (m_Oplus * nu_Oplus_1)
#    print('sigma_Oplus.shape =', sigma_Oplus_1.shape)
#    
#    sigma_NOplus_1 = (n_NOplus_1 * e**2) / (m_NOplus * nu_NOplus_1)
#    
#    sigma_N2plus_1 = (n_N2plus_1 * e**2) / (m_N2plus * nu_N2plus_1)
#    
#    sigma_O2plus_1 = (n_O2plus_1 * e**2) / (m_O2plus * nu_O2plus_1)
#    
#    sigma_Nplus_1 = (n_Nplus_1 * e**2) / (m_Nplus * nu_Nplus_1)
#    
#    sigma_Hplus_1 = (n_Hplus_1 * e**2) / (m_Hplus * nu_Hplus_1)
#    
#    sigma_ion_total_1 = (sigma_Oplus_1 + sigma_NOplus_1 + sigma_N2plus_1 + sigma_O2plus_1 +
#                       sigma_Nplus_1 + sigma_Hplus_1)
#        
#    sigma_ion_1 = [sigma_Oplus_1, sigma_NOplus_1, sigma_N2plus_1, sigma_O2plus_1, sigma_Nplus_1, sigma_Hplus_1]
#    
#    # Electron conductivity
#    
#    #sigma_e = (ne * e**2) / (mass_e * elec_coll_freq) (Schunk and Nagy Eq 5.115)
#    # elec_coll_freq is the collision frequency of the electrons summed over all
#    # neutral species
#    
#    nu_en_O_1 = 8.9e-11 * (n_O_1 * 1e-6) * (1.0 + 5.7e-4 * Te_1) * np.sqrt(Te_1)
#    print('nu_en_O.shape =', nu_en_O_1.shape)
#    
#    nu_en_N2_1 = 2.33e-11 * (n_N2_1 * 1e-6) * (1.0 - 1.21e-4 * Te_1) * Te_1
#    
#    nu_en_O2_1 = 1.82e-10 * (n_O2_1 * 1e-6) * (1.0 + 3.6e-2 * np.sqrt(Te_1)) * np.sqrt(Te_1)
#    
#    nu_en_H_1 = 4.5e-9 * (n_H_1 * 1e-6) * (1.0 - 1.35e-4 * Te_1) * np.sqrt(Te_1)
#    
#    nu_en_total_1 = nu_en_O_1 + nu_en_N2_1 + nu_en_O2_1 + nu_en_H_1
#    
#    sigma_e_1 = (ne_1 * e**2) / (m_e * nu_en_total_1)
#        #print('sigma_e =', sigma_e)
#    
#    # Pedersen conductivity (S&N equation 5.119)
#    
#    sigma_P_1 = np.zeros(np.shape(ne_1))
#    
#    for i in np.arange(0,6,1):
#        sigma_P_1 += (sigma_ion_1[i] * (nu_ion_1[i]**2 / (nu_ion_1[i]**2 + 
#        (e * B / mass_ions[i])**2)))
#    
#    print('sigma_P shape =', sigma_P_1.shape)
#    
#    
#    # Hall conductivity (S&N equation 5.120)
#    
#    sigma_H_ion_1 = np.zeros(np.shape(ne_1))
#    
#    for j in range(0,6,1):
#        sigma_H_ion_1 += (sigma_ion_1[j] * ((nu_ion_1[j] * ((e * B) / mass_ions[j]) / (nu_ion_1[j]**2 + 
#        (e * B / mass_ions[j])**2)))) 
#        #print('Hall conductivity =', sigma_H)
#        
#        
#    sigma_H_1 = -sigma_H_ion_1 + ((nu_en_total_1 * sigma_e_1) / ((e * B) / m_e))
#        
#    print('sigma_H shape =', sigma_H_1.shape)
#        
#    # Parallel Conductivity (S&N equation 5.125a)
#    
#    # Nu_parallel = summation of electron-ion collisions + summation of electron-neutral collisions
#    
#    # Electron-ion (nu_ei).  Need to loop through all ion species.  Use Schunk and Nagy
#    # Equation 4.144.
#    
#    nu_ei_1 = np.zeros(np.shape(ne_1))
#    
#    for k in np.arange(0, 6, 1):
#        nu_ei_1 += 54.5 * ((ion_dens_1[k] * 1e-6) / Te_1**1.5)
#        #print('nu_ei =', nu_ei)
#        
#    # Electron-neutral interactions (Schunk and Nagy Table 4.6)
#    
#    nu_para_1 = nu_ei_1 + nu_en_total_1
#    
#    sigma_para_1 = (ne_1 * e**2) / (m_e * nu_para_1)
#    print('sigma_para_1 shape =', sigma_para_1.shape)
#        
#    sigma_perp_1 = sigma_P_1 + sigma_H_1
#    print('sigma_perp_1 shape =', sigma_perp_1.shape)
#
#    return sigma_para_1, sigma_perp_1

    return sigma_P, sigma_H


### PLOTTING STUFF???
#plt.figure(dpi=250)
#sigma_H_plot = np.transpose(sigma_H_1[iz,:,:],[1,0])
#plt.pcolormesh(x/1e3, y/1e3, sigma_H_plot * 1e7, cmap=plt.cm.get_cmap('coolwarm'))
#plt.xlim([-950, -50])
#plt.ylim([-450, 450])
#plt.xticks([-950, -725, -500, -275, -50], fontsize=8)
#plt.yticks([-450, -225, 0, 225, 450], fontsize=8)
#plt.text(-935, -435, ('5:20:00'), fontsize=4)
#plt.xlabel("x (km)", fontsize=9)
#plt.ylabel("y (km)", fontsize=9)
##plt.clim(-0.3, 0.3)
#cbar = plt.colorbar() #ticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
#cbar.ax.yaxis.offsetText.set(size=8)
#cbar.ax.set_ylabel('Hall Conductivity (x10'r'$^{-7}$' + ' ' + 'S/m'')', fontsize=8)
#cbar.ax.tick_params(labelsize=8)
#plt.show()
#
#plt.figure(dpi=250)
#sigma_P_plot = np.transpose(sigma_P_1[iz,:,:],[1,0])
#plt.pcolormesh(x/1e3, y/1e3, sigma_P_plot, cmap=plt.cm.get_cmap('coolwarm'))
#plt.xlim([-950, -50])
#plt.ylim([-450, 450])
#plt.xticks([-950, -725, -500, -275, -50], fontsize=8)
#plt.yticks([-450, -225, 0, 225, 450], fontsize=8)
#plt.text(-935, -435, ('5:20:00'), fontsize=4)
#plt.xlabel("x (km)", fontsize=9)
#plt.ylabel("y (km)", fontsize=9)
##plt.clim(-0.3, 0.3)
#cbar = plt.colorbar() #ticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
#cbar.ax.yaxis.offsetText.set(size=8)
#cbar.ax.set_ylabel('Pedersen Conductivity (x10'r'$^{-7}$' + ' ' + 'S/m'')', fontsize=8)
#cbar.ax.tick_params(labelsize=8)
#plt.show()
#
#plt.figure(dpi=250)
#sigma_Para_plot = np.transpose(sigma_para_1[iz,:,:],[1,0])
#plt.pcolormesh(x/1e3, y/1e3, sigma_Para_plot, cmap=plt.cm.get_cmap('coolwarm'))
#plt.xlim([-950, -50])
#plt.ylim([-450, 450])
#plt.xticks([-950, -725, -500, -275, -50], fontsize=8)
#plt.yticks([-450, -225, 0, 225, 450], fontsize=8)
#plt.text(-935, -435, ('5:20:00'), fontsize=4)
#plt.xlabel("x (km)", fontsize=9)
#plt.ylabel("y (km)", fontsize=9)
##plt.clim(-0.3, 0.3)
#cbar = plt.colorbar() #ticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
#cbar.ax.yaxis.offsetText.set(size=8)
#cbar.ax.set_ylabel('Parallel Conductivity (x10'r'$^{-7}$' + ' ' + 'S/m'')', fontsize=8)
#cbar.ax.tick_params(labelsize=8)
#plt.show()




#mapfact_P = np.sqrt(sigma_para / sigma_P)
#mapfact_total = np.sqrt(sigma_para / sigma_perp)
#lambda_para = 50 * mapfact_P
#print('lambda_para =', mapfact_P[:,525,70])
#print('z =', z/1e3)
#np.savetxt('Altitude_values', z/1e3)

# sigma_P_pt1_1 = []
# sigma_P_pt2_1 = []
# sigma_P_pt3_1 = []
# sigma_P_pt4_1 = []

# sigma_H_pt1_1 = []
# sigma_H_pt2_1 = []
# sigma_H_pt3_1 = []
# sigma_H_pt4_1 = []

# sigma_para_pt1_1 = []
# sigma_para_pt2_1 = []
# sigma_para_pt3_1 = []
# sigma_para_pt4_1 = []

# sigma_perp_pt1_1 = []
# sigma_perp_pt2_1 = []
# sigma_perp_pt3_1 = []
# sigma_perp_pt4_1 = []

# for s in np.arange(0, z.size, 1):

#     Ppt1_1 = sigma_P_1[s,273,727]
#     sigma_P_pt1_1.append(Ppt1_1)

#     Ppt2_1 = sigma_P_1[s,663,727]
#     sigma_P_pt2_1.append(Ppt2_1)

#     Ppt3_1 = sigma_P_1[s,273,531]
#     sigma_P_pt3_1.append(Ppt3_1)

#     Ppt4_1 = sigma_P_1[s,663,531]
#     sigma_P_pt4_1.append(Ppt4_1)
    
#     Hpt1_1 = sigma_H_1[s,273,727]
#     sigma_H_pt1_1.append(Hpt1_1)
    
#     Hpt2_1 = sigma_H_1[s,663,727]
#     sigma_H_pt2_1.append(Hpt2_1)
    
#     Hpt3_1 = sigma_H_1[s,273,531]
#     sigma_H_pt3_1.append(Hpt3_1)
    
#     Hpt4_1 = sigma_H_1[s,663,531]
#     sigma_H_pt4_1.append(Hpt4_1)
# z_low_res = z
# print('z_low_res =', z_low_res)

# direc2 = "/Volumes/Elements_1/GDI_RISR_Test_Cases/18_49_High_Res_Vertical/Disturb/"
# #direc2 = "/Volumes/Elements/GDI_RISR_Test_Cases/18_49_High_Res_Vertical_BGPhi/Disturb/"
# data = gemini3d.read.frame(direc2, datetime(2017,11,21,18,59,00), var=["ne", "Te", "Ti", "ns", "Ts"]) 
# print(data.keys())
# print(data)

# cfg = gemini3d.read.config(direc2)
# times = cfg["time"]

# xg = gemini3d.read.grid(direc2)

# ne = np.array(data["ne"])
# Te = np.array(data["Te"])
# Ti = np.array(data["Ti"])
# ns = np. array(data["ns"])
# Ts = np.array(data["Ts"])

# z = np.array(xg["x1"][2:-2])
# x = np.array(xg["x2"][2:-2])
# y = np.array(xg["x3"][2:-2])
# print('z =', z)
    
# n_Oplus = ns[0,:,:,:]
# n_NOplus = ns[1,:,:,:]
# n_N2plus = ns[2,:,:,:]
# n_O2plus = ns[3,:,:,:]
# n_Nplus = ns[4,:,:,:]
# n_Hplus = ns[5,:,:,:]
# ion_dens = [n_Oplus, n_NOplus, n_N2plus, n_O2plus, n_Nplus, n_Hplus]
    
# T_Oplus = Ts[0,:,:,:]
# T_NOplus = Ts[1,:,:,:]
# T_N2plus = Ts[2,:,:,:]
# T_O2plus = Ts[3,:,:,:]
# T_Nplus = Ts[4,:,:,:]
# T_Hplus = Ts[5,:,:,:]
# Te = Ts[6,:,:,:]
# ion_temp = [T_Oplus, T_NOplus, T_N2plus, T_O2plus, T_Nplus, T_Hplus]

# # Ion conductivity (Schunk and Nagy equation 5.112)

# # sigma_ion = (n_ion * e_ion**2) / (mass_ion * ion-netrals_coll_freq) (Schunk and Nagy Eq 5.112) 
# # ion-neutrals_coll_freq is the collision frequency of each individual ion summed over
# # neutral species

# os.environ["GEMINI_ROOT"] = "~/libgem/"
# msisdata = gemini3d.msis.msis_setup(cfg,xg)
# #print(msisdata.keys())
    
# n_O = np.array(msisdata["nO"])
# print('n_O.shape =', n_O.shape)
# n_N2 = np.array(msisdata["nN2"])
# n_O2 = np.array(msisdata["nO2"])
# n_N = np.array(msisdata["nN"])
# n_H = np.array(msisdata["nH"])
# Tn = np.array(msisdata["Tn"])
    
# ntrl_dens = [n_O, n_N2, n_O2, n_N, n_H] 

# # For individual ion-neutral pairs, the non-resonant collision frequency is given by
# # S&N equation (4.146): nu_in_nonres = C_in * n_n.  C_in values are listed in 
# # Table 4.4.  n_n is in units of cm^-3.
  
# nu_in_nonres_Oplus = (6.82e-10 * (n_N2 *1e-6) + 6.64e-10 * (n_O2 *1e-6) + 
#                       4.62e-10 * (n_N * 1e-6))

# nu_in_nonres_NOplus = (2.44e-10 * (n_O * 1e-6) + 4.34e-10 * (n_N2 * 1e-6) + 
#                        4.27e-10 * (n_O2 * 1e-6) + 2.79e-10 * (n_N * 1e-6) + 
#                        0.69e-10 * (n_H * 1e-6))

# nu_in_nonres_N2plus = (2.58e-10 * (n_O * 1e-6) + 4.49e-10 * (n_O2 * 1e-6) + 
#                        2.95e-10 * (n_N * 1e-6) + 0.74e-10 * (n_H * 1e-6))

# nu_in_nonres_O2plus = (2.31e-10 * (n_O * 1e-6) + 4.13e-10 * (n_N2 * 1e-6) + 
#                        2.64e-10 * (n_N * 1e-6) + 0.65e-10 * (n_H * 1e-6))

# nu_in_nonres_Nplus = (4.42e-10 * (n_O * 1e-6) + 7.47e-10 * (n_N2 * 1e-6) + 
#                       7.25e-10 * (n_O2 * 1e-6) + 1.45e-10 * (n_H * 1e-6))

# nu_in_nonres_Hplus = (33.6e-10 * (n_N2 * 1e-6) + 32.0e-10 * (n_O2 * 1e-6) + 
#                       26.1e-10 * (n_N * 1e-6))

# nu_in_nonres_total = (nu_in_nonres_Oplus + nu_in_nonres_NOplus + nu_in_nonres_N2plus +
#     nu_in_nonres_O2plus + nu_in_nonres_Nplus + nu_in_nonres_Hplus)

# # For individual ion-neutral pairs, the resonant collision frequencies are given by the 
# # equations in Table 4.5

# #breakpoint()
# nu_in_res_OplusO = 3.67e-11 * (n_O * 1e-6) * np.sqrt((T_Oplus + Tn) / 2) * (1.0 - 0.064 * np.log10((T_Oplus + Tn) / 2))**2

# nu_in_res_OplusH = 4.63e-12 * (n_H * 1e-6) * np.sqrt((Tn + T_Oplus) / 16)

# nu_in_res_N2plusN2 = 5.14e-11 * (n_N2 * 1e-6) * np.sqrt((T_N2plus + Tn) / 2) * (1.0 - 0.069 * np.log10((T_N2plus + Tn) / 2))**2

# nu_in_res_O2plusO2 = 2.59e-11 * (n_O2 * 1e-6) * np.sqrt((T_O2plus + Tn) / 2) * (1.0 - 0.073 * np.log10((T_O2plus + Tn) / 2))**2

# nu_in_res_NplusN = 3.83e-11 * (n_N * 1e-6) * np.sqrt((T_Nplus + Tn)/2) * (1.0 - 0.063 * np.log10((T_Nplus + Tn) / 2))**2

# nu_in_res_HplusH = 2.65e-10 * (n_H * 1e-6) * np.sqrt((T_Hplus + Tn) / 2) * (1.0 - 0.083 * np.log10((T_Hplus + Tn) / 2))**2

# nu_in_res_HplusO = 6.61e-11 * (n_O * 1e-6) * np.sqrt(T_Hplus) * (1.0 - 0.047 * np.log10(T_Hplus))**2

# nu_in_res_total = (nu_in_res_OplusO + nu_in_res_OplusH + nu_in_res_N2plusN2 + nu_in_res_O2plusO2 + 
#                    nu_in_res_NplusN + nu_in_res_HplusH)

# nu_Oplus = nu_in_nonres_Oplus + nu_in_res_OplusO + nu_in_res_OplusH
    
# nu_NOplus = nu_in_nonres_NOplus
    
# nu_N2plus = nu_in_nonres_N2plus + nu_in_res_N2plusN2
    
# nu_O2plus =  nu_in_nonres_O2plus + nu_in_res_O2plusO2
    
# nu_Nplus = nu_in_nonres_Nplus + nu_in_res_NplusN
    
# nu_Hplus = nu_in_nonres_Hplus + nu_in_res_HplusH + nu_in_res_HplusO
    
# nu_ion = [nu_Oplus, nu_NOplus, nu_N2plus, nu_O2plus, nu_Nplus, nu_Hplus]

# sigma_Oplus = (n_Oplus * e**2) / (m_Oplus * nu_Oplus)
# print('sigma_Oplus.shape =', sigma_Oplus.shape)

# sigma_NOplus = (n_NOplus * e**2) / (m_NOplus * nu_NOplus)

# sigma_N2plus = (n_N2plus * e**2) / (m_N2plus * nu_N2plus)

# sigma_O2plus = (n_O2plus * e**2) / (m_O2plus * nu_O2plus)

# sigma_Nplus = (n_Nplus * e**2) / (m_Nplus * nu_Nplus)

# sigma_Hplus = (n_Hplus * e**2) / (m_Hplus * nu_Hplus)

# sigma_ion_total = (sigma_Oplus + sigma_NOplus + sigma_N2plus + sigma_O2plus +
#                    sigma_Nplus + sigma_Hplus)
    
# sigma_ion = [sigma_Oplus, sigma_NOplus, sigma_N2plus, sigma_O2plus, sigma_Nplus, sigma_Hplus]

# # Electron conductivity

# #sigma_e = (ne * e**2) / (mass_e * elec_coll_freq) (Schunk and Nagy Eq 5.115)
# # elec_coll_freq is the collision frequency of the electrons summed over all
# # neutral species

# nu_en_O = 8.9e-11 * (n_O * 1e-6) * (1.0 + 5.7e-4 * Te) * np.sqrt(Te)
# print('nu_en_O.shape =', nu_en_O.shape)

# nu_en_N2 = 2.33e-11 * (n_N2 * 1e-6) * (1.0 - 1.21e-4 * Te) * Te

# nu_en_O2 = 1.82e-10 * (n_O2 * 1e-6) * (1.0 + 3.6e-2 * np.sqrt(Te)) * np.sqrt(Te)

# nu_en_H = 4.5e-9 * (n_H * 1e-6) * (1.0 - 1.35e-4 * Te) * np.sqrt(Te)

# nu_en_total = nu_en_O + nu_en_N2 + nu_en_O2 + nu_en_H

# sigma_e = (ne * e**2) / (m_e * nu_en_total)

# # Pedersen conductivity (S&N equation 5.119)

# sigma_P = np.zeros(np.shape(ne))

# for i in np.arange(0,6,1):
#     sigma_P += (sigma_ion[i] * (nu_ion[i]**2 / (nu_ion[i]**2 + 
#     (e * B / mass_ions[i])**2)))

# print('sigma_P.shape =', sigma_P.shape)


# # Hall conductivity (S&N equation 5.120)

# sigma_H_ion = np.zeros(np.shape(ne))

# for j in range(0,6,1):
#     sigma_H_ion += (sigma_ion[j] * ((nu_ion[j] * ((e * B) / mass_ions[j]) / (nu_ion[j]**2 + 
#     (e * B / mass_ions[j])**2)))) 
    
    
# sigma_H = -sigma_H_ion + ((nu_en_total * sigma_e) / ((e * B) / m_e))
    
# print('sigma_H.shape =', sigma_H.shape)
    
# # Parallel Conductivity (S&N equation 5.125a)

# # Nu_parallel = summation of electron-ion collisions + summation of electron-neutral collisions

# # Electron-ion (nu_ei).  Need to loop through all ion species.  Use Schunk and Nagy
# # Equation 4.144.

# nu_ei = np.zeros(np.shape(ne))

# for k in np.arange(0, 6, 1):
#     nu_ei += 54.5 * ((ion_dens[k] * 1e-6) / Te**1.5)
#     #print('nu_ei =', nu_ei)
    
# # Electron-neutral interactions (Schunk and Nagy Table 4.6)

# nu_para = nu_ei + nu_en_total

# sigma_para = (ne * e**2) / (m_e * nu_para)
    
# sigma_perp = sigma_P + sigma_H
# #mapfact_P = np.sqrt(sigma_para / sigma_P)
# #mapfact_total = np.sqrt(sigma_para / sigma_perp)
# #lambda_para = 50 * mapfact_P
# #print('lambda_para =', mapfact_P[:,525,70])

# sigma_P_pt1 = []
# sigma_P_pt2 = []
# sigma_P_pt3 = []
# sigma_P_pt4 = []

# sigma_H_pt1 = []
# sigma_H_pt2 = []
# sigma_H_pt3 = []
# sigma_H_pt4 = []

# sigma_para_pt1 = []
# sigma_para_pt2 = []
# sigma_para_pt3 = []
# sigma_para_pt4 = []

# sigma_perp_pt1 = []
# sigma_perp_pt2 = []
# sigma_perp_pt3 = []
# sigma_perp_pt4 = []

# for s in np.arange(0, z.size, 1):
#     plt.figure(dpi=250)
    
#     Ppt1 = sigma_P[s,273,727]
#     sigma_P_pt1.append(Ppt1)
    
#     Ppt2 = sigma_P[s,663,727]
#     sigma_P_pt2.append(Ppt2)
    
#     Ppt3 = sigma_P[s,273,531]
#     sigma_P_pt3.append(Ppt3)
    
#     Ppt4 = sigma_P[s,663,531]
#     sigma_P_pt4.append(Ppt4)
    
    
#     Hpt1 = sigma_H[s,273,727]
#     sigma_H_pt1.append(Hpt1)
    
#     Hpt2 = sigma_H[s,663,727]
#     sigma_H_pt2.append(Hpt2)
    
#     Hpt3 = sigma_H[s,273,531]
#     sigma_H_pt3.append(Hpt3)
    
#     Hpt4 = sigma_H[s,663,531]
#     sigma_H_pt4.append(Hpt4)
    
# print('sigma_P_pt1 =', sigma_P_pt1)

# print('sigma_P_pt1_1 =', sigma_P_pt1_1)

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt1_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# # plt.plot(sigma_P_pt1, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-9, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Pedersen Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#1, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt1_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt1, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1.0e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Pedersen Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(0.25e-6, 52, 'Pt#1, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt2_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt2, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-9, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Pedersen Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#2, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt2_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt2, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1.0e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Pedersen Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(0.25e-6, 52, 'Pt#2, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt3_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt3, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-9, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Pedersen Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#3, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt3_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt3, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1.0e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Pedersen Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(0.25e-6, 52, 'Pt#3, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt4_1, z_low_res/1e3, 'b-.') #, label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt4, z/1e3, 'r:', label='High Vertical Resolution')
# #plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-9, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Pederdsen Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#4, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_P_pt4_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_P_pt4, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1.0e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Pedersen Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(0.25e-6, 52, 'Pt#4, 10+00, PhiWBG=1e-3', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt1_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt1, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-13, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Hall Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#1, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt1_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt1, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Hall Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(8.5e-6, 52, 'Pt#1, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt2_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt2, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-13, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Hall Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#2, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt2_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt2, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Hall Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(8.5e-6, 52, 'Pt#2, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt3_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt3, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-13, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Hall Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#3, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt3_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt3, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Hall Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(8.5e-6, 52, 'Pt#3, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt4_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt4, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.xscale("log")
# plt.xlim(1e-13, 1e-5)
# plt.ylim(50, 350)
# plt.ylabel("Altitude (km)")
# plt.xlabel("Hall Conductivity", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# plt.text(1e-6, 55, 'Pt#4, 25+00', fontsize=4)
# plt.show()

# plt.figure(dpi=250)
# plt.plot(sigma_H_pt4_1, z_low_res/1e3, 'b-.', label='Low Vertical Resolution')
# #plt.plot(sigma_H_pt4, z/1e3, 'r:', label='High Vertical Resolution')
# plt.legend(loc='upper right', fontsize=7)
# plt.ylim(50, 150)
# plt.xlim(0, 1e-5)
# #plt.yticks([1e-9, 1e-8, 1e-7, 1e-6], fontsize=8)
# plt.ylabel("Altitude (km)", fontsize=8)
# plt.xlabel("Hall Conductivity (Linear x-axis)", fontsize=8)
# # plt.xticks([1e-13, 1e-10, 1e-7, 1e-4, 1e-1, 1e2], fontsize=8)
# # plt.yticks([75, 125, 175, 225, 275], fontsize=8)
# # plt.legend(loc='upper right', fontsize=7)
# # plt.title('Pedersen, Hall, Perpendicular (Hall + Pedersen) & Parallel Conductivities (Point #4)', fontsize=7)
# # plt.text(1e-1, 55, 'Elapsed Time: 15:00', fontsize=5)
# # plt.text(1.4e-13, 55, '18_49_Vert_Disturb', fontsize=5)
# plt.text(8.5e-6, 52, 'Pt#4, 25+00', fontsize=4)
# plt.show()


