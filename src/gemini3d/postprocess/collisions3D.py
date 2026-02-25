"""
Calculate collision frequencies for full gemini grid

Equations from Shunk and Nagy, 2009

Original script - M. Redden, 2022
M. Redden, 2023
"""



import gemini3d.read
import gemini3d.msis
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os as os
import xarray as xr


def collisionfrequency(
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



    #kb = 1.38e-23
    #amu = 1.66e-27
    #e = 1.60e-19
    #B = 5.0e-5 #Teslas
    #m_e = 9.11e-31
    #m_Oplus = 2.21e-27
    #m_NOplus = 4.15e-27
    #m_N2plus = 3.87e-27
    #m_O2plus = 4.43e-27
    #m_Nplus = 1.94e-27
    #m_Hplus = 1.38e-28
    #mass_ions = [m_Oplus, m_NOplus, m_N2plus, m_O2plus, m_Nplus, m_Hplus]
    #mass_ions_total = m_Oplus + m_NOplus + m_N2plus + m_O2plus + m_Nplus + m_Hplus
    
    # O = 16, N2 = 28, O2 = 32, N = 14, NO = 30, H =1 
    
    
    ##direc = "/Volumes/Elements_2/Simulation_10/"
    #direc  = "~/Desktop/2nd_Paper_Stuff/Round_grad_scale/"
    #cfg = gemini3d.read.config(direc)
    cfg = gemini3d.read.config(path)
    #print(cfg)
    #times = cfg["time"]
    
    #xg = gemini3d.read.grid(direc)
    xg = gemini3d.read.grid(path)
    #print(xg)
    
    #dat = gemini3d.read.frame(direc, datetime(2013,2,20,5,80,00), var=["ne", "Te", "Ti", "ns", "Ts"]) 
    dat = gemini3d.read.frame(path, time, var=["ne", "Te", "Ti", "ns", "Ts"]) 
    #dat = gemini3d.read.frame(path, time) 
    #print(dat["ns"])

    print('DAT DIMENSIONS')
    print(dat)

    #dat_mod = dat.rename_dims({'x1':'alt_km', 'x2':'glat', 'x3':'glon'})
    ns = dat["ns"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])
    Ts = dat["Ts"].assign_coords(species=["O+", "NO+", "N2+", "O2+", "N+", "H+", "e"])
    

    #print(Ts)
    
    #print(dat_mod)
    ne_1 = np.array(dat["ne"])
    #print('ne.shape =', ne.shape)
    Te_1 = np.array(dat["Te"])
    Ti_1 = np.array(dat["Ti"])
    ns_1 = np. array(dat["ns"])
    Ts_1 = np.array(dat["Ts"])


    
    #z = np.array(xg["x1"][2:-2])
    #x = np.array(xg["x2"][2:-2])
    #y = np.array(xg["x3"][2:-2])
    #
    #refalt = 120e3
    #iz = np.argmin(abs(z-refalt))     # find the index where altitude is 
    #        #closest to reference altitude
    #print('iz = ', iz)

    # ns = [O+, NO+, N2+, O2+, N+, H+]
        
    n_Oplus_1 = ns_1[0,:,:,:]
    n_NOplus_1 = ns_1[1,:,:,:]
    n_N2plus_1 = ns_1[2,:,:,:]
    n_O2plus_1 = ns_1[3,:,:,:]
    n_Nplus_1 = ns_1[4,:,:,:]
    n_Hplus_1 = ns_1[5,:,:,:]
    ion_dens_1 = [n_Oplus_1, n_NOplus_1, n_N2plus_1, n_O2plus_1, n_Nplus_1, n_Hplus_1]
        
    T_Oplus_1 = Ts_1[0,:,:,:]
    T_NOplus_1 = Ts_1[1,:,:,:]
    T_N2plus_1 = Ts_1[2,:,:,:]
    T_O2plus_1 = Ts_1[3,:,:,:]
    T_Nplus_1 = Ts_1[4,:,:,:]
    T_Hplus_1 = Ts_1[5,:,:,:]
    Te_1 = Ts_1[6,:,:,:]
    ion_temp_1 = [T_Oplus_1, T_NOplus_1, T_N2plus_1, T_O2plus_1, T_Nplus_1, T_Hplus_1]
    
    # Ion conductivity (Schunk and Nagy equation 5.112)
    
    # sigma_ion = (n_ion * e_ion**2) / (mass_ion * ion-netrals_coll_freq) (Schunk and Nagy Eq 5.112) 
    # ion-neutrals_coll_freq is the collision frequency of each individual ion summed over
    # neutral species
    
    #os.environ["GEMINI_ROOT"] = "~/libgem/"
    msisdata0 = gemini3d.msis.msis_setup(cfg,xg)

    #msisdata.value
    print(list(msisdata0.keys()))

#    breakpoint()
#
#
#    msisdata1 = msisdata.rename_dims(alt_km='x1', glat='x2', glon='x3')
#    msisdata2 = msisdata1.assign_coords(x1=(msisdata1.x1*1000.))
#    print('MSISDATA')
#    print(msisdata2)
#        
#    n_O_1 = np.array(msisdata["nO"])
#    n_N2_1 = np.array(msisdata["nN2"])
#    n_O2_1 = np.array(msisdata["nO2"])
#    n_N_1 = np.array(msisdata["nN"])
#    n_H_1 = np.array(msisdata["nH"])
#    Tn_1 = np.array(msisdata["Tn"])
#
#
#    msisdat = xr.Dataset(coords={"alt_km": alt1, "glat": glat1, "glon": glon1})
#
#        for k in {"nO", "nN2", "nO2", "Tn", "nN", "nH"}:
#            atmos[k] = (("alt_km", "glat", "glon"), f[f"/{k}"][:])



    msisdata = xr.DataArray(dims=['x1','x2','x3'], coords=[dat.x1, dat.x2, dat.x3])
    for k in list(msisdata0.keys()):
        msisdata[k] = (('x1','x2','x3'), np.array(msisdata0[k]))


    print(msisdata)
        
    #ntrl_dens_1 = [n_O_1, n_N2_1, n_O2_1, n_N_1, n_H_1] 
    
    # For individual ion-neutral pairs, the non-resonant collision frequency is given by
    # S&N equation (4.146): nu_in_nonres = C_in * n_n.  C_in values are listed in 
    # Table 4.4.  n_n is in units of cm^-3.


    #nu_in = xr.Dataset(data_vars={"O+":(["N2","O2","N"],[blank,blank,blank),
    #                              "N+":(["N2","O2","N"],[blank,blank,blank)})
    nu_in = xr.DataArray(dims=["ion", "neutral", "x1", "x2", "x3"], coords=(["O+", "NO+", "N2+", "O2+", "N+", "H+"], 
                               ["O", "N2", "O2", "N", "H"], dat["x1"], dat["x2"], dat["x3"]))

    #print(nu_in)
    #print(nu_in.loc["O+","O"])


    # O+
    nu_in.loc["O+","N2"] = 6.82e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["O+","O2"] = 6.64e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["O+","N"]  = 4.62e-10 * msisdata["nN"]  * 1e-6
      
    #nu_in_nonres_Oplus_1 = (6.82e-10 * (n_N2_1 *1e-6) + 6.64e-10 * (n_O2_1 *1e-6) + 
    #                      4.62e-10 * (n_N_1 * 1e-6))
    
    # NO+
    nu_in.loc["NO+","O"]  = 2.44e-10 * msisdata["nO"]  * 1e-6
    nu_in.loc["NO+","N2"] = 4.34e-10 * msisdata["nN2"] * 1e-6 
    nu_in.loc["NO+","O2"] = 4.27e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["NO+","N"]  = 2.79e-10 * msisdata["nN"]  * 1e-6
    nu_in.loc["NO+","H"]  = 0.69e-10 * msisdata["nH"]  * 1e-6

    #nu_in_nonres_NOplus_1 = (2.44e-10 * (n_O_1 * 1e-6) + 4.34e-10 * (n_N2_1 * 1e-6) + 
    #                       4.27e-10 * (n_O2_1 * 1e-6) + 2.79e-10 * (n_N_1 * 1e-6) + 
    #                       0.69e-10 * (n_H_1 * 1e-6))
   
    # N2+
    nu_in.loc["N2+","O"]  = 2.58e-10 * msisdata["nO"]  * 1e-6
    nu_in.loc["N2+","O2"] = 4.49e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["N2+","N"]  = 2.95e-10 * msisdata["nN"]  * 1e-6
    nu_in.loc["N2+","H"]  = 0.74e-10 * msisdata["nH"]  * 1e-6

    #nu_in_nonres_N2plus_1 = (2.58e-10 * (n_O_1 * 1e-6) + 4.49e-10 * (n_O2_1 * 1e-6) + 
    #                       2.95e-10 * (n_N_1 * 1e-6) + 0.74e-10 * (n_H_1 * 1e-6))
   
    # O2+
    nu_in.loc["N2+","O"]  = 2.31e-10 * msisdata["nO"]  * 1e-6
    nu_in.loc["N2+","N2"] = 4.13e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["N2+","N"]  = 2.64e-10 * msisdata["nN"]  * 1e-6
    nu_in.loc["N2+","H"]  = 0.65e-10 * msisdata["nH"]  * 1e-6

    #nu_in_nonres_O2plus_1 = (2.31e-10 * (n_O_1 * 1e-6) + 4.13e-10 * (n_N2_1 * 1e-6) + 
    #                       2.64e-10 * (n_N_1 * 1e-6) + 0.65e-10 * (n_H_1 * 1e-6))
   
    # N+
    nu_in.loc["N+","O"]  = 4.42e-10 * msisdata["nO"]  * 1e-6
    nu_in.loc["N+","N2"] = 7.47e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["N+","O2"] = 7.25e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["N+","O"]  = 1.45e-10 * msisdata["nH"]  * 1e-6

    #nu_in_nonres_Nplus_1 = (4.42e-10 * (n_O_1 * 1e-6) + 7.47e-10 * (n_N2_1 * 1e-6) + 
    #                      7.25e-10 * (n_O2_1 * 1e-6) + 1.45e-10 * (n_H_1 * 1e-6))
    
    # H+
    nu_in.loc["H+","N2"] = 33.6e-10 * msisdata["nN2"] * 1e-6
    nu_in.loc["H+","O2"] = 32.0e-10 * msisdata["nO2"] * 1e-6
    nu_in.loc["H+","N"]  = 26.1e-10 * msisdata["nN"]  * 1e-6

    #nu_in_nonres_Hplus_1 = (33.6e-10 * (n_N2_1 * 1e-6) + 32.0e-10 * (n_O2_1 * 1e-6) + 
    #                      26.1e-10 * (n_N_1 * 1e-6))
    
    #####
    # DO WE ACTUALLY NEED TO CALCULATE TOTAL NONRESOSONANT NU??
    #####

    #nu_in_nonres_total_1 = (nu_in_nonres_Oplus_1 + nu_in_nonres_NOplus_1 + nu_in_nonres_N2plus_1 +
    #    nu_in_nonres_O2plus_1 + nu_in_nonres_Nplus_1 + nu_in_nonres_Hplus_1)
    
    # For individual ion-neutral pairs, the resonant collision frequencies are given by the 
    # equations in Table 4.5
    
    #breakpoint()
    #print(Ts.keys())

    #print(Ts.sel(species="O+"))

    # Ts and msisdata have different dimension names - makes this fail
    #Tr = (np.array(Ts.sel(species="O+")) + np.array(msisdata["Tn"])) / 2
    Tr = (Ts.sel(species="O+") + msisdata["Tn"]) / 2
    #print('Tr')
    #print(Tr)
    nu_in.loc["O+","O"] = 3.67e-11 * (msisdata["nO"] * 1e-6) * np.sqrt(Tr) * (1.0 - 0.064 * np.log10(Tr))**2
    #nu_in_res_OplusO_1 = 3.67e-11 * (n_O_1 * 1e-6) * np.sqrt((T_Oplus_1 + Tn_1) / 2) * (1.0 - 0.064 * np.log10((T_Oplus_1 + Tn_1) / 2))**2
   

    nu_in.loc["O+","H"] = 4.63e-12 * (msisdata["nH"] * 1e-6) * np.sqrt((msisdata["Tn"] + Ts.sel(species="O+")) / 16)
    #nu_in_res_OplusH_1 = 4.63e-12 * (n_H_1 * 1e-6) * np.sqrt((Tn_1 + T_Oplus_1) / 16)
    
    Tr = (Ts.sel(species="N2+") + msisdata["Tn"]) / 2
    nu_in.loc["N2+","N2"] = 5.14e-11 * (msisdata["nN2"] * 1e-6) * np.sqrt(Tr) * (1.0 - 0.069 * np.log10(Tr))**2
    #nu_in_res_N2plusN2_1 = 5.14e-11 * (n_N2_1 * 1e-6) * np.sqrt((T_N2plus_1 + Tn_1) / 2) * (1.0 - 0.069 * np.log10((T_N2plus_1 + Tn_1) / 2))**2
    
    Tr = (Ts.sel(species="O2+") + msisdata["Tn"]) / 2
    nu_in.loc["O2+","O2"] = 2.59e-11 * (msisdata["nO2"] * 1e-6) * np.sqrt(Tr) * (1.0 - 0.073 * np.log10(Tr))**2
    #nu_in_res_O2plusO2_1 = 2.59e-11 * (n_O2_1 * 1e-6) * np.sqrt((T_O2plus_1 + Tn_1) / 2) * (1.0 - 0.073 * np.log10((T_O2plus_1 + Tn_1) / 2))**2
    
    Tr = (Ts.sel(species="N+") + msisdata["Tn"]) / 2
    nu_in.loc["N+","N"] = 3.83e-11 * (msisdata["nN"] * 1e-6) * np.sqrt(Tr) * (1.0 - 0.063 * np.log10(Tr))**2
    #nu_in_res_NplusN_1 = 3.83e-11 * (n_N_1 * 1e-6) * np.sqrt((T_Nplus_1 + Tn_1)/2) * (1.0 - 0.063 * np.log10((T_Nplus_1 + Tn_1) / 2))**2
    
    Tr = (Ts.sel(species="H+") + msisdata["Tn"]) / 2
    nu_in.loc["H+","H"] = 2.65e-10 * (msisdata["nH"] * 1e-6) * np.sqrt(Tr) * (1.0 - 0.083 * np.log10(Tr))**2
    #nu_in_res_HplusH_1 = 2.65e-10 * (n_H_1 * 1e-6) * np.sqrt((T_Hplus_1 + Tn_1) / 2) * (1.0 - 0.083 * np.log10((T_Hplus_1 + Tn_1) / 2))**2
    
    nu_in.loc["H+","O"] = 6.61e-11 * (msisdata["nO"] * 1e-6) * np.sqrt(Ts.sel(species="H+")) * (1.0 - 0.047 * np.log10(Ts.sel(species="H+")))**2
    #nu_in_res_HplusO_1 = 6.61e-11 * (n_O_1 * 1e-6) * np.sqrt(T_Hplus_1) * (1.0 - 0.047 * np.log10(T_Hplus_1))**2
    

    #print(nu_in)

    #nu_in_res_total_1 = (nu_in_res_OplusO_1 + nu_in_res_OplusH_1 + nu_in_res_N2plusN2_1 + nu_in_res_O2plusO2_1 + 
    #                   nu_in_res_NplusN_1 + nu_in_res_HplusH_1)
    #
    #nu_Oplus_1 = nu_in_nonres_Oplus_1 + nu_in_res_OplusO_1 + nu_in_res_OplusH_1
    #    
    #nu_NOplus_1 = nu_in_nonres_NOplus_1
    #    
    #nu_N2plus_1 = nu_in_nonres_N2plus_1 + nu_in_res_N2plusN2_1
    #    
    #nu_O2plus_1 =  nu_in_nonres_O2plus_1 + nu_in_res_O2plusO2_1
    #    
    #nu_Nplus_1 = nu_in_nonres_Nplus_1 + nu_in_res_NplusN_1
    #    
    #nu_Hplus_1 = nu_in_nonres_Hplus_1 + nu_in_res_HplusH_1 + nu_in_res_HplusO_1
    #    
    #nu_ion_1 = [nu_Oplus_1, nu_NOplus_1, nu_N2plus_1, nu_O2plus_1, nu_Nplus_1, nu_Hplus_1]


    # Total ion-neutral collision frequency
    # sum all ion and neutral dimensions of nu_in
    nu_in_tot = nu_in.sum(dim=("ion","neutral"))
    #print(nu_in_tot)
    




    #sigma_Oplus_1 = (n_Oplus_1 * e**2) / (m_Oplus * nu_Oplus_1)
    #print('sigma_Oplus.shape =', sigma_Oplus_1.shape)
    #
    #sigma_NOplus_1 = (n_NOplus_1 * e**2) / (m_NOplus * nu_NOplus_1)
    #
    #sigma_N2plus_1 = (n_N2plus_1 * e**2) / (m_N2plus * nu_N2plus_1)
    #
    #sigma_O2plus_1 = (n_O2plus_1 * e**2) / (m_O2plus * nu_O2plus_1)
    #
    #sigma_Nplus_1 = (n_Nplus_1 * e**2) / (m_Nplus * nu_Nplus_1)
    #
    #sigma_Hplus_1 = (n_Hplus_1 * e**2) / (m_Hplus * nu_Hplus_1)
    #
    #sigma_ion_total_1 = (sigma_Oplus_1 + sigma_NOplus_1 + sigma_N2plus_1 + sigma_O2plus_1 +
    #                   sigma_Nplus_1 + sigma_Hplus_1)
    #    
    #sigma_ion_1 = [sigma_Oplus_1, sigma_NOplus_1, sigma_N2plus_1, sigma_O2plus_1, sigma_Nplus_1, sigma_Hplus_1]
    
    # Electron conductivity
    
    #sigma_e = (ne * e**2) / (mass_e * elec_coll_freq) (Schunk and Nagy Eq 5.115)
    # elec_coll_freq is the collision frequency of the electrons summed over all
    # neutral species


    #nu_en = xr.DataArray(dims=["neutral", "alt_km", "glat", "glon"], coords=(["O", "N2", "O2", "N", "H"], msisdata["alt_km"], msisdata["glat"], msisdata["glon"]))

    nu_en = xr.DataArray(dims=["neutral", "x1", "x2", "x3"], coords=(["O", "N2", "O2", "N", "H"], dat["x1"], dat["x2"], dat["x3"]))

    
    nu_en.loc["O"] = 8.9e-11 * (msisdata["nO"] * 1e-6) * (1.0 + 5.7e-4 * dat["Te"]) * np.sqrt(dat["Te"])
    #nu_en_O_1 = 8.9e-11 * (n_O_1 * 1e-6) * (1.0 + 5.7e-4 * Te_1) * np.sqrt(Te_1)
    #print('nu_en_O.shape =', nu_en_O_1.shape)
    
    nu_en.loc["N2"] = 2.33e-11 * (msisdata["nN2"] * 1e-6) * (1.0 - 1.21e-4 * dat["Te"]) * dat["Te"]
    #nu_en_N2_1 = 2.33e-11 * (n_N2_1 * 1e-6) * (1.0 - 1.21e-4 * Te_1) * Te_1
    
    nu_en.loc["O2"] = 1.82e-10 * (msisdata["nO2"] * 1e-6) * (1.0 + 3.6e-2 * np.sqrt(dat["Te"])) * np.sqrt(dat["Te"])
    #nu_en_O2_1 = 1.82e-10 * (n_O2_1 * 1e-6) * (1.0 + 3.6e-2 * np.sqrt(Te_1)) * np.sqrt(Te_1)
    
    nu_en.loc["H"] = 4.5e-9 * (msisdata["nH"] * 1e-6) * (1.0 - 1.35e-4 * dat["Te"]) * np.sqrt(dat["Te"])
    #nu_en_H_1 = 4.5e-9 * (n_H_1 * 1e-6) * (1.0 - 1.35e-4 * Te_1) * np.sqrt(Te_1)
    
    #nu_en_total_1 = nu_en_O_1 + nu_en_N2_1 + nu_en_O2_1 + nu_en_H_1

    ### Total Electron-Neutral Collision Frequency
    # Sum over neutral dimension of nu_en
    nu_en_tot = nu_en.sum(dim='neutral')
    #print(nu_en_tot)

    
    #sigma_e_1 = (ne_1 * e**2) / (m_e * nu_en_total_1)
        #print('sigma_e =', sigma_e)
    
    ## Pedersen conductivity (S&N equation 5.119)
    #
    #sigma_P_1 = np.zeros(np.shape(ne_1))
    #
    #for i in np.arange(0,6,1):
    #    sigma_P_1 += (sigma_ion_1[i] * (nu_ion_1[i]**2 / (nu_ion_1[i]**2 + 
    #    (e * B / mass_ions[i])**2)))
    #
    #print('sigma_P shape =', sigma_P_1.shape)
    #
    #
    ## Hall conductivity (S&N equation 5.120)
    #
    #sigma_H_ion_1 = np.zeros(np.shape(ne_1))
    #
    #for j in range(0,6,1):
    #    sigma_H_ion_1 += (sigma_ion_1[j] * ((nu_ion_1[j] * ((e * B) / mass_ions[j]) / (nu_ion_1[j]**2 + 
    #    (e * B / mass_ions[j])**2)))) 
    #    #print('Hall conductivity =', sigma_H)
    #    
    #    
    #sigma_H_1 = -sigma_H_ion_1 + ((nu_en_total_1 * sigma_e_1) / ((e * B) / m_e))
    #    
    #print('sigma_H shape =', sigma_H_1.shape)
        
    # Parallel Conductivity (S&N equation 5.125a)
    
    # Nu_parallel = summation of electron-ion collisions + summation of electron-neutral collisions
    
    # Electron-ion (nu_ei).  Need to loop through all ion species.  Use Schunk and Nagy
    # Equation 4.144.
    
    nu_ei = xr.DataArray(dims=["ion", "x1", "x2", "x3"], coords=(["O+", "NO+", "N2+", "O2+", "N+", "H+"], dat["x1"], dat["x2"], dat["x3"]))

    #Ts.sel(species="O+")
    print(nu_ei.coords["ion"].values)
    for s in nu_ei.coords["ion"].values:
        nu_ei.loc[s] = 54.5 * ((Ts.sel(species=s) * 1e-6) / dat["Te"]**1.5)

    #nu_ei_1 = np.zeros(np.shape(ne_1))
    #
    #for k in np.arange(0, 6, 1):
    #    nu_ei_1 += 54.5 * ((ion_dens_1[k] * 1e-6) / Te_1**1.5)
    #    #print('nu_ei =', nu_ei)
        
    ## Electron-neutral interactions (Schunk and Nagy Table 4.6)
    #
    #nu_para_1 = nu_ei_1 + nu_en_total_1
    #
    #sigma_para_1 = (ne_1 * e**2) / (m_e * nu_para_1)
    #print('sigma_para_1 shape =', sigma_para_1.shape)
    #    
    #sigma_perp_1 = sigma_P_1 + sigma_H_1
    #print('sigma_perp_1 shape =', sigma_perp_1.shape)

    # Types of collisions
    # - Ion-Neutral [NixNn] X
    #   - Ion-Netural resonant X
    # - Ion-Ion [NixNi]
    # - Electron-Neutral [Nn] X
    # - Electron-Ion [Ni]
    # - ion collision frequency
    # - electron collision frequency

    #return sigma_para_1, sigma_perp_1

    return nu_in, nu_en, nu_in_tot, nu_en_tot

